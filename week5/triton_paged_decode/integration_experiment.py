"""Correctness first, then unprofiled paired timings. No persistent JIT cache.

Run from any directory. Outputs JSON to stdout; model weights are local-only.
"""
import gc
import json
import os
from pathlib import Path
import statistics as stats
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nanovllm_qwen35'))


def main():
    import torch
    from flash_attn import flash_attn_with_kvcache
    from qwen35_adapter.triton_paged import paged_decode_trusted
    from paged_decode import paged_decode
    from verify import reference
    from qwen35_adapter.gpu_engine import Qwen35Engine
    from nanovllm.sampling_params import SamplingParams
    import qwen35_adapter.pool_backend as backend_module
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
              'micro':[], 'correctness':[], 'engine':[]}

    def emit(tag, data):
        print(json.dumps({'stage':tag, 'data':data}), flush=True)

    def errors(a,b):
        a,b=a.float(),b.float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        return {'max_abs':(a-b).abs().max().item(),
                'relative_l2':((a-b).norm()/a.norm().clamp_min(1e-12)).item()}

    # Micro: same quantized QKV, noncontiguous disjoint pages, poison invalid KV.
    # Event intervals include GPU idle time if the host cannot submit fast enough.
    for b,n in ((1,128),(1,1024),(4,128),(4,1024)):
        page=256; width=(n+page-1)//page; npages=b*width+2
        q=torch.randn(b,16,256,device='cuda',dtype=torch.bfloat16)
        k=torch.full((npages,page,4,256),float('nan'),device='cuda',dtype=q.dtype)
        v=torch.full_like(k,float('nan'))
        tables=torch.randperm(npages,device='cuda')[:b*width].reshape(b,width).int().contiguous()
        lengths=torch.full((b,),n,device='cuda',dtype=torch.int32)
        t=torch.arange(n,device='cuda')
        slots=tables[:,t//page].long()*page+t%page
        k.view(-1,4,256)[slots]=torch.randn(b,n,4,256,device='cuda',dtype=q.dtype)*.3
        v.view(-1,4,256)[slots]=torch.randn(b,n,4,256,device='cuda',dtype=q.dtype)
        expected=reference(q,k,v,tables,lengths).to(q.device,q.dtype)
        def dense():
            kk=k.view(-1,4,256)[slots].transpose(1,2).repeat_interleave(4,1)
            vv=v.view(-1,4,256)[slots].transpose(1,2).repeat_interleave(4,1)
            p=((q[:,:,None,:]@kk.transpose(-1,-2))*(256**-.5)).float().softmax(-1).to(q.dtype)
            return (p@vv)[:,:,0,:]
        funcs={'dense':dense,
               'flash_paged':lambda:flash_attn_with_kvcache(q[:,None],k,v,
                   cache_seqlens=lengths,block_table=tables,causal=True)[:,0]}
        for tile in (8,16,32):
            funcs[f'triton_{tile}']=lambda tile=tile:paged_decode_trusted(q,k,v,tables,lengths,block_t=tile)
        records={}
        kbytes=k.view(torch.uint8).clone();vbytes=v.view(torch.uint8).clone()
        for name,fn in funcs.items():
            result=fn()
            torch.testing.assert_close(result,expected,atol=.008,rtol=.01)
            if name.startswith('triton'):
                torch.testing.assert_close(result,paged_decode(q,k,v,tables,lengths,
                    block_t=int(name.split('_')[1])),atol=0,rtol=0)
            records[name]={'error':errors(expected,result),'event_ms':[], 'wall_ms':[]}
            for _ in range(10):fn()
        assert torch.equal(k.view(torch.uint8),kbytes) and torch.equal(v.view(torch.uint8),vbytes)
        for rep in range(5):
            names=list(funcs)
            if rep%2:names.reverse()
            for name in names:
                torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();begin.record()
                for _ in range(50):funcs[name]()
                end.record();end.synchronize()
                records[name]['wall_ms'].append((time.perf_counter()-wall)*1000/50)
                records[name]['event_ms'].append(begin.elapsed_time(end)/50)
        if os.environ.get('TRITON_DEVICE_TIMING') == '1':
            # Measurement only: capture 16 independent calls to remove Python
            # gaps between calls. This does NOT graph-enable the model/engine.
            for name,fn in funcs.items():
                graph=torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    for _ in range(16):fn()
                for _ in range(2):graph.replay()
                torch.cuda.synchronize()
                values=[]
                for _ in range(5):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();graph.replay();end.record();end.synchronize()
                    values.append(begin.elapsed_time(end)/16)
                records[name]['graph_amortized_ms']=values
                del graph
        case={'B':b,'L':n,'records':records}
        report['micro'].append(case);emit('micro',case)
    del q,k,v,tables,lengths,expected,result,kbytes,vbytes,funcs,records
    gc.collect();torch.cuda.empty_cache()

    if os.environ.get('TRITON_MICRO_ONLY') == '1':
        emit('complete_micro',report)
        return
    engine=Qwen35Engine('/home/ubuntu/huggingface/Qwen3.5-4B',max_model_len=512,
                        max_num_seqs=4,decode_backend='dense',lm_head_mode='row')
    seed=engine.tokenizer('Explain how KV cache improves language model inference.',
                          add_special_tokens=False)['input_ids']
    def prompts(b,n):
        return [(seed*((n+b)//len(seed)+2))[i:i+n] for i in range(b)]

    # Fixed-token teacher forcing: every route sees exactly the same prefix.
    # Separate diagnostic run: per-layer outputs, logical KV, conv/recurrent/logits.
    original=backend_module.decoder_layer
    def diagnostic(mode,ps):
        engine.backend.decode_backend=mode
        seqs=[engine.add_request(p,SamplingParams(max_tokens=4,ignore_eos=True)) for p in ps]
        traces=[]
        for step in range(4):
            batch=engine.scheduler.schedule();inputs=engine.runner.builder.prepare(batch)
            layers=[]
            def capture(*args,**kwargs):
                out,new=original(*args,**kwargs)
                entry={'layer':args[4],'out':out.detach().cpu()}
                if hasattr(new,'conv'):
                    entry.update(conv=new.conv.detach().cpu(),recurrent=new.recurrent.detach().cpu())
                layers.append(entry)
                return out,new
            backend_module.decoder_layer=capture
            try:hidden=engine.backend.forward(inputs)
            finally:backend_module.decoder_layer=original
            logits=engine.backend.compute_logits(hidden[inputs.last_indices.to(hidden.device)]).cpu()
            logical=[]
            for row,n in enumerate(inputs.context_lens.tolist()):
                t=torch.arange(n);slots=(inputs.block_tables[row,t//256].long()*256+t%256).to('cuda')
                logical.append(engine.kv_pool.flatten(2,3).index_select(2,slots).cpu())
            traces.append({'layers':layers,'logits':logits,'kv':logical})
            engine.states.synchronize()
            # Valid IDs, same across backends; not claiming these are greedy outputs.
            engine.scheduler.postprocess(batch,tuple(seed[(step+i)%len(seed)] for i in range(len(seqs))))
        assert engine.scheduler.is_finished()
        return traces
    try:
        for ps in (prompts(2,19),prompts(2,257)):
            ref=diagnostic('dense',ps)
            for mode in ('flash_paged','triton_paged'):
                trial=diagnostic(mode,ps)
                worst={};first=None;argmax_equal=True
                for step,(a,b) in enumerate(zip(ref,trial)):
                    argmax_equal &= torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1))
                    pairs=[('logits',a['logits'],b['logits'])]
                    pairs += [('kv',x,y) for x,y in zip(a['kv'],b['kv'])]
                    for j,(x,y) in enumerate(zip(a['layers'],b['layers'])):
                        assert x['layer']==y['layer']
                        for key in ('out','conv','recurrent'):
                            if key in x:pairs.append((key,x[key],y[key]))
                        if first is None and not torch.equal(x['out'],y['out']):
                            first={'step':step,'layer':x['layer'],'request_row':j%2}
                    for name,x,y in pairs:
                        err=errors(x,y)
                        prev=worst.setdefault(name,{'max_abs':0.,'relative_l2':0.})
                        for metric in err:prev[metric]=max(prev[metric],err[metric])
                        # Explicit exploratory guard, not a model-quality guarantee.
                        if err['relative_l2']>.03:
                            raise AssertionError(f'{mode} {step} {name} relative_l2 > 3%: {err}')
                case={'L':len(ps[0]),'B':2,'mode':mode,'worst':worst,
                      'first_non_bitwise_layer':first,'all_logits_argmax_equal':argmax_equal}
                report['correctness'].append(case);emit('correctness',case)
                del trial
            del ref
        gc.collect();torch.cuda.empty_cache()

        # Real generation, no snapshots/profiler. Keep LM Head row mode constant.
        def generation(mode,b,n):
            engine.backend.decode_backend=mode
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            seqs=[engine.add_request(p,SamplingParams(max_tokens=8,ignore_eos=True)) for p in prompts(b,n)]
            steps=[]
            while not engine.scheduler.is_finished():
                before=time.perf_counter();result=engine.step();now=time.perf_counter()
                steps.append((result.is_prefill,(now-before)*1000))
                if result.is_prefill:first=now
            elapsed=now-start
            return {'ttft_ms':(first-start)*1000,'tpot_ms':(now-first)*1000/7,
                'output_tps':b*8/elapsed,'decode_tps':b*7/(now-first),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'ids':[list(s.completion_token_ids) for s in seqs]}
        modes=['dense','flash_paged','triton_paged']
        for b,n in ((1,128),(4,128),(1,256),(4,256)):
            for mode in modes:
                for _ in range(2):generation(mode,b,n)
            runs={mode:[] for mode in modes}
            for rep in range(6):
                order=modes[rep%3:]+modes[:rep%3]
                if rep>=3:order=list(reversed(order))
                for mode in order:runs[mode].append(generation(mode,b,n))
                emit('engine_progress',{'B':b,'L':n,'repeat':rep+1})
            ids=runs['dense'][0]['ids']
            same_ids=all(r['ids']==ids for rs in runs.values() for r in rs)
            case={'B':b,'L':n,'all_generated_ids_equal':same_ids,'runs':runs}
            report['engine'].append(case);emit('engine',case)
        emit('complete',report)
    finally:
        backend_module.decoder_layer=original
        if not engine.runner.poisoned and not engine.states.poisoned and engine.scheduler._pending is None:
            engine.close()


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='infer-triton-integration-') as cache:
        os.environ['TRITON_CACHE_DIR']=cache
        main()
