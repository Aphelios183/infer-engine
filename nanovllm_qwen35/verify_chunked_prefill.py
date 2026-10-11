"""Bounded executable checks, no pytest tree. Control protocol + real GPU state alignment."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from qwen35_adapter.state_manager import StateManager
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.scheduler import HybridScheduler
from qwen35_adapter.runner_inputs import RunnerInputBuilder
from qwen35_adapter.execution import CPUExecutionRunner


def controls():
    states = StateManager(torch.zeros(1,2,1,2),torch.zeros(1,2,1,1,1))
    scheduler = HybridScheduler(AdmissionController(BlockManager(8,4),states,max_prefill_tokens=32),
                                max_num_seqs=2,token_budget=6,eos_ids={99},prefill_chunk_size=3)
    def add(n,limit=2):
        seq=Sequence(list(range(1,n+1)),SamplingParams(max_tokens=limit,ignore_eos=True))
        seq.block_size=4; scheduler.add(seq); return seq
    b,a=add(7),add(3)
    batch=scheduler.schedule(); inputs=RunnerInputBuilder(scheduler).prepare(batch)
    assert batch.is_prefill and batch.input_counts==(3,3) and batch.sample_rows==(1,)
    assert inputs.last_indices.tolist()==[5] and inputs.sample_rows==(1,)
    assert len(b.block_table)==2  # reserve the entire 7-token prompt on first chunk
    slot=states.lookup(b.seq_id); states.conv_pool[:,slot].fill_(7)
    a.temperature=0.7;b.temperature=1.3
    class FinalBackend:
        execution_mode='cpu_sync'
        def forward(self, inp):return torch.zeros(len(inp.input_ids),2)
        def compute_logits(self, hidden):
            assert hidden.shape==(1,2);return torch.zeros(1,32)
    class FinalSampler:
        execution_mode='cpu_sync'
        def sample(self,logits,temperatures):
            assert temperatures==(0.7,);return torch.tensor([20],dtype=torch.int64)
    first=CPUExecutionRunner(scheduler,FinalBackend(),FinalSampler()).run(batch)
    assert first.sample_rows==(1,) and first.token_ids==(20,)
    assert (b.num_tokens,b.num_cached_tokens,a.num_tokens,a.num_cached_tokens)==(7,3,4,3)
    batch=scheduler.schedule()
    assert not batch.is_prefill and batch.sequences==(a,)  # never mix partial B with decode A
    scheduler.postprocess(batch,[21]); assert a.is_finished
    batch=scheduler.schedule()
    assert batch.sequences==(b,) and batch.input_counts==(3,) and batch.sample_rows==()
    assert states.lookup(b.seq_id)==slot and torch.all(states.conv_pool[:,slot]==7)
    class Backend:
        execution_mode='cpu_sync'
        def forward(self, inp): return torch.zeros(len(inp.input_ids),2)
        def compute_logits(self, hidden): raise AssertionError('intermediate chunk called LM Head')
    class Sampler:
        execution_mode='cpu_sync'
        def sample(self,*args): raise AssertionError('intermediate chunk called sampler')
    result=CPUExecutionRunner(scheduler,Backend(),Sampler()).run(batch)
    assert result.token_ids==() and result.sample_rows==() and b.num_cached_tokens==6 and b.num_tokens==7
    batch=scheduler.schedule(); assert batch.input_counts==(1,) and batch.sample_rows==(0,)
    scheduler.postprocess(batch,[22]); assert b.num_cached_tokens==7 and b.num_tokens==8
    scheduler.cancel(b.seq_id); assert not states.owners
    # Slot reuse is initialized; a pending middle chunk cannot be cancelled unsafely.
    c=add(9); batch=scheduler.schedule()
    assert torch.count_nonzero(states.conv_pool[:,states.lookup(c.seq_id)])==0
    try: scheduler.cancel(c.seq_id)
    except RuntimeError: pass
    else: raise AssertionError('cancelled pending execution')
    scheduler.postprocess(batch,[]); scheduler.cancel(c.seq_id)
    # A stray output for an intermediate chunk fails and releases, never appends.
    d=add(9); batch=scheduler.schedule()
    try: scheduler.postprocess(batch,[3])
    except ValueError: pass
    else: raise AssertionError('accepted a spurious token')
    assert d.is_finished and not states.owners and len(scheduler.admission.blocks.free_block_ids)==8
    zero=add(9,0); assert zero.is_finished and not states.owners
    # Per-request cap is distinct from the total batch budget.
    scheduler.prefill_chunk_size=4;scheduler.token_budget=5
    e,f=add(9,1),add(5,1)
    rounds=[]
    while not scheduler.is_finished():
        batch=scheduler.schedule();RunnerInputBuilder(scheduler).prepare(batch)
        assert batch.total_tokens<=5 and max(batch.input_counts)<=4
        rounds.append(batch.input_counts)
        scheduler.postprocess(batch,[20]*len(batch.sample_rows))
    assert rounds==[(4,1),(4,1),(1,3)] and e.is_finished and f.is_finished
    return {'passed':True,'cases':['sample-row mapping','no mixed batch','no intermediate sampling',
            'full prompt block reservation','continued state retention','pending cancel guard',
            'partial cancel/reuse','invalid output disposal','zero budget','clipped global budget']}


def metrics(actual,expected):
    a,b=actual.float(),expected.float()
    return {'max_abs':(a-b).abs().max().item(),
            'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}


@torch.inference_mode()
def gpu_checks(model, report):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,Qwen3_5TextRotaryEmbedding
    from qwen35_adapter.gpu_engine import Qwen35Engine
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Qwen35Engine(model,max_model_len=512,max_num_seqs=2,block_size=4,
                        prefill_chunk_size=3,max_num_batched_tokens=6)
    try:
        config=Qwen3_5TextConfig(**engine.parameters.config['text_config']);config._attn_implementation='eager'
        with torch.device('meta'): reference=Qwen3_5ForCausalLM(config)
        reference.load_state_dict(engine.parameters.state_dict(),strict=True,assign=True)
        reference.tie_weights();reference.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config,device='cuda:0')
        reference.eval();engine.runner.capture_logits=True
        base=engine.tokenizer('Explain the purpose of a key value cache.',add_special_tokens=False)['input_ids']
        def ids(n): return (base*(n//len(base)+1))[:n]
        report['gpu_steps']=[];report['full_vs_chunked']=[]
        # Last sample row may be 1, empty, or all. Exercise 1-token tails and block boundaries.
        for cap,lengths in ((3,[7,3]),(1,[3]),(4,[9,5]),(4,[5,6]),(16,[33,17]),(64,[65]),(64,[128])):
            engine.scheduler.prefill_chunk_size=cap;engine.scheduler.token_budget=cap*2
            seqs=[engine.add_request(ids(n),SamplingParams(max_tokens=3,ignore_eos=True)) for n in lengths]
            caches={s.seq_id:DynamicCache(config=config) for s in seqs}
            step_number=0
            while not engine.scheduler.is_finished():
                if step_number==1 and len(engine.scheduler.running)>1:
                    engine.scheduler.running.rotate(1)  # changing batch row cannot change state ownership
                batch=engine.scheduler.schedule();inp=RunnerInputBuilder(engine.scheduler).prepare(batch)
                before={s.seq_id:(s.num_tokens,s.num_cached_tokens) for s in batch.sequences}
                expected=[]
                for row,s in enumerate(batch.sequences):
                    a,b=inp.cu_seqlens_q[row:row+2].tolist()
                    ref=reference(input_ids=inp.input_ids[a:b][None].cuda(),
                                  position_ids=inp.positions[a:b][None].cuda(),
                                  past_key_values=caches[s.seq_id],use_cache=True,logits_to_keep=1)
                    if row in inp.sample_rows: expected.append(ref.logits[0,-1].cpu())
                with patch.object(engine.backend,'compute_logits',wraps=engine.backend.compute_logits) as head:
                    result=engine.runner.run(batch)
                assert head.call_count==bool(inp.sample_rows)
                if expected:
                    expected=torch.stack(expected)
                    error=metrics(engine.runner.last_logits,expected)
                    if not torch.equal(engine.runner.last_logits,expected):
                        raise AssertionError(f'same-segmentation HF logits differ: {error}')
                    assert result.token_ids==tuple(expected.argmax(-1).tolist())
                else:
                    assert result.token_ids==() and engine.runner.last_logits is None
                for row,s in enumerate(batch.sequences):
                    slot=inp.state_slots[row].item();n=inp.context_lens[row].item()
                    for layer_id,refstate in enumerate(caches[s.seq_id].layers):
                        if layer_id in engine.backend.linear_map:
                            idx=engine.backend.linear_map[layer_id]
                            assert torch.equal(engine.states.conv_pool[idx,slot],refstate.conv_states[0]),f'conv layer {layer_id}'
                            assert torch.equal(engine.states.recurrent_pool[idx,slot],refstate.recurrent_states[0]),f'recurrent layer {layer_id}'
                        else:
                            idx=engine.backend.full_map[layer_id];logical=torch.arange(n)
                            slots=(inp.block_tables[row,logical//4].long()*4+logical%4).cuda()
                            for side,t in enumerate((refstate.keys,refstate.values)):
                                kv=engine.kv_pool[side,idx].view(-1,4,256).index_select(0,slots)
                                assert torch.equal(kv,t[0].transpose(0,1)),f'KV layer {layer_id}'
                    old_tokens,old_cached=before[s.seq_id]
                    assert s.num_tokens==old_tokens+(row in inp.sample_rows)
                    processed=engine.scheduler.completed[s.seq_id].processed_tokens if s.is_finished else s.num_cached_tokens
                    assert processed==old_cached+batch.input_counts[row]
                    # Compare complete-prefill arithmetic too, but do not silently relax exactness.
                    if batch.is_prefill and row in inp.sample_rows:
                        full=reference(input_ids=torch.tensor([s.prompt_token_ids],device='cuda:0'),use_cache=True,logits_to_keep=1)
                        local_index=inp.sample_rows.index(row)
                        comparison=metrics(engine.runner.last_logits[local_index],full.logits[0,-1].cpu())
                        maxima={'kv':0.,'conv':0.,'recurrent':0.}
                        for lid,(seg,whole) in enumerate(zip(caches[s.seq_id].layers,full.past_key_values.layers)):
                            fields=(('conv','conv_states'),('recurrent','recurrent_states')) if lid in engine.backend.linear_map else (('kv','keys'),('kv','values'))
                            for kind,name in fields:
                                maxima[kind]=max(maxima[kind],(getattr(seg,name).float()-getattr(whole,name).float()).abs().max().item())
                        report['full_vs_chunked'].append({'cap':cap,'length':s.num_prompt_tokens,**comparison,'state_max_abs':maxima})
                record={'cap':cap,'lengths':lengths,'prefill':batch.is_prefill,
                        'counts':batch.input_counts,'sample_rows':inp.sample_rows,'last_indices':inp.last_indices.tolist()}
                report['gpu_steps'].append(record);print(json.dumps(record),flush=True)
                step_number+=1
            assert not engine.states.owners and not engine.scheduler.admission.blocks.used_block_ids
        # A failing forward in a no-sample chunk still synchronizes and releases.
        engine.scheduler.prefill_chunk_size=2;engine.scheduler.token_budget=4
        seq=engine.add_request(ids(7),SamplingParams(max_tokens=2))
        original=engine.backend.forward
        def fail_after_write(inp): original(inp);raise RuntimeError('chunk failure after state writes')
        with patch.object(engine.backend,'forward',side_effect=fail_after_write):
            try:engine.step()
            except RuntimeError as exc:assert 'chunk failure' in str(exc)
            else:raise AssertionError('missing failure')
        assert seq.is_finished and not engine.states.owners and engine.scheduler._pending is None
        # Partial cancellation -> dirty slot reused by another fresh request.
        seq=engine.add_request(ids(7));engine.step();assert seq.num_cached_tokens==2
        engine.scheduler.cancel(seq.seq_id);assert not engine.states.owners
        seq=engine.add_request(ids(3),SamplingParams(max_tokens=1,ignore_eos=True))
        while not engine.scheduler.is_finished():engine.step()
        assert len(seq.completion_token_ids)==1 and not engine.states.owners
        # EOS applies only to generated output on the final chunk.
        original_head=engine.backend.compute_logits;eos=next(iter(engine.scheduler.eos_ids))
        def force_eos(hidden):
            logits=original_head(hidden);logits.fill_(-100);logits[:,eos]=100;return logits
        with patch.object(engine.backend,'compute_logits',side_effect=force_eos):
            out=engine.generate([ids(7)],SamplingParams(max_tokens=4))
        assert out[0]['stop_reason']=='eos' and out[0]['token_ids']==[eos]
        report['same_segmentation_exact']=True
        report['partial_failure_cancel_reuse_eos']=True
    finally:
        if not engine.runner.poisoned and not engine.states.poisoned:
            if engine.scheduler._pending is not None:
                engine.states.synchronize()
                engine.scheduler.fail_batch(engine.scheduler._pending,'verification interrupted',execution_complete=True)
            engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--model',default='/home/ubuntu/huggingface/Qwen3.5-4B')
    p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise RuntimeError('Choose a new output path')
    report={'status':'running'}
    try:
        report['control']=controls()
        if not args.cpu_only:gpu_checks(args.model,report)
        report['status']='complete'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:output.write_text(json.dumps(report,indent=2))
