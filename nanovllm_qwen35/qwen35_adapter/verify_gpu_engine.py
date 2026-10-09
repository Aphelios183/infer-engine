"""Real 4B GPU engine validation, bounded workloads; not a speed benchmark."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,Qwen3_5TextRotaryEmbedding
from nanovllm.sampling_params import SamplingParams
from .gpu_engine import Qwen35Engine
from .runner_inputs import RunnerInputBuilder


@torch.inference_mode()
def verify(model_dir,output):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Qwen35Engine(model_dir,max_model_len=512,max_num_seqs=2,block_size=64)
    c=Qwen3_5TextConfig(**engine.parameters.config['text_config']);c._attn_implementation='eager'
    with torch.device('meta'):reference=Qwen3_5ForCausalLM(c)
    reference.load_state_dict(engine.parameters.state_dict(),strict=True,assign=True)
    reference.tie_weights();reference.model.rotary_emb=Qwen3_5TextRotaryEmbedding(c,device='cuda:0')
    reference.eval() # shared read-only weights, independent calculation and cache
    engine.runner.capture_logits=True
    base=engine.tokenizer('请解释KV缓存。',add_special_tokens=False)['input_ids']
    records=[]
    def ids(length):return (base*((length+len(base)-1)//len(base)))[:length]
    for lengths in ([1],[3],[19],[63],[64],[65],[128],[257],[63,65]):
        seqs=[engine.add_request(ids(n),SamplingParams(max_tokens=3,ignore_eos=True)) for n in lengths]
        caches={s.seq_id:DynamicCache(config=c) for s in seqs}
        steps=0
        while not engine.scheduler.is_finished():
            # Deliberately reverse running order after prefill to exercise slot ownership.
            if steps==1 and len(seqs)>1:engine.scheduler.running.rotate(1)
            batch=engine.scheduler.schedule()
            inp=RunnerInputBuilder(engine.scheduler).prepare(batch)
            expected=[]
            for row,s in enumerate(batch.sequences):
                a,b=inp.cu_seqlens_q[row:row+2].tolist()
                out=reference(input_ids=inp.input_ids[a:b][None].cuda(),
                              position_ids=inp.positions[a:b][None].cuda(),
                              past_key_values=caches[s.seq_id],use_cache=True,logits_to_keep=1)
                expected.append(out.logits[0,-1].cpu())
            result=engine.runner.run(batch)
            expected=torch.stack(expected);actual=engine.runner.last_logits
            error=(actual.float()-expected.float()).abs().max().item()
            same=result.token_ids==tuple(expected.argmax(-1).tolist())
            item={'lengths':lengths,'step':steps,'max_abs':error,'tokens_equal':same}
            records.append(item);print(json.dumps(item),flush=True)
            if not torch.equal(actual,expected):
                raise AssertionError('exact BF16 logits comparison failed; inspect before relaxing tolerance')
            assert same
            # Compare the original pool slices, even on the terminal step (dirty bytes retained).
            for row,s in enumerate(batch.sequences):
                slot=inp.state_slots[row].item();n=inp.context_lens[row].item()
                for layer_id,refstate in enumerate(caches[s.seq_id].layers):
                    if layer_id in engine.backend.linear_map:
                        idx=engine.backend.linear_map[layer_id]
                        assert torch.equal(engine.states.conv_pool[idx,slot],refstate.conv_states[0])
                        assert torch.equal(engine.states.recurrent_pool[idx,slot],refstate.recurrent_states[0])
                    else:
                        idx=engine.backend.full_map[layer_id]
                        logical=torch.arange(n)
                        slots=(inp.block_tables[row,logical//64].long()*64+logical%64).cuda()
                        for side,t in enumerate((refstate.keys,refstate.values)):
                            actual_kv=engine.kv_pool[side,idx].view(-1,4,256).index_select(0,slots)
                            assert torch.equal(actual_kv,t[0].transpose(0,1))
            steps+=1
        assert not engine.states.owners
        assert len(engine.scheduler.admission.blocks.free_block_ids)==len(engine.scheduler.admission.blocks.blocks)
    # EOS path through runner, after real forward; forced logits only test stopping semantics.
    original=engine.backend.compute_logits
    eos=next(iter(engine.scheduler.eos_ids))
    def force_eos(x):
        logits=original(x);logits.fill_(-100);logits[:,eos]=100;return logits
    with patch.object(engine.backend,'compute_logits',side_effect=force_eos):
        stopped=engine.generate([ids(3)],SamplingParams(max_tokens=5))
    assert stopped[0]['stop_reason']=='eos' and len(stopped[0]['token_ids'])==1
    zero=engine.generate([ids(3)],SamplingParams(max_tokens=0))
    assert zero[0]['token_ids']==[]
    # Recoverable host-side failure after all state writes: synchronize before disposal.
    with patch.object(engine.backend,'compute_logits',side_effect=RuntimeError('injected host failure')):
        try:engine.generate([ids(3)],SamplingParams(max_tokens=2))
        except RuntimeError as exc:assert 'injected' in str(exc)
        else:raise AssertionError('failure injection not detected')
    assert engine.scheduler.is_finished() and not engine.states.owners
    rendered=engine.tokenizer.apply_chat_template([{'role':'user','content':'请用一句话说明KV缓存的作用。'}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
    text=engine.generate([rendered],SamplingParams(max_tokens=32))
    torch.cuda.synchronize()
    report={'device':torch.cuda.get_device_name(0),'reference':'original HF torch chunk, eager, same checkpoint dtype',
            'records':records,'exact_pool_state_comparisons':True,'forced_eos_pass':True,
            'zero_budget_pass':True,'host_failure_recovery_pass':True,'text_generation':text,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'text_generation':text,'cases':len(records)},ensure_ascii=False),flush=True)
    engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.model_dir,a.output)
