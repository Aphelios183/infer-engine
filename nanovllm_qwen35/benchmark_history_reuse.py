"""History-address optimization: isolated profiling, exact A/B, unprofiled paired timing."""
import argparse,json,time,statistics,types
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from nanovllm import SamplingParams
from qwen35_adapter.gpu_engine import Qwen35Engine
from qwen35_adapter import pool_backend as pb,layer_compute as lc
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

def wrap(name,fn):
    def call(*a,**kw):
        with record_function(name):return fn(*a,**kw)
    return call

@torch.inference_mode()
def main(args):
    out=Path(args.output);out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    e=Qwen35Engine('/home/ubuntu/huggingface/Qwen3.5-4B',max_model_len=256,max_num_seqs=2,block_size=16,prefill_chunk_size=64)
    original=e.backend.forward
    ns=pb.__dict__.copy()
    exec(compile(Path(args.before).read_text(),args.before,'exec'),ns)
    before=types.MethodType(ns['PooledCPUBackend'].forward,e.backend)
    variants={'before':before,'after':original}
    base=e.tokenizer('Explain the purpose of a key value cache.',add_special_tokens=False)['input_ids']
    ids=(base*20)[:128]
    def begin(fn):
        e.backend.forward=fn
        s=e.add_request(ids,SamplingParams(max_tokens=1,ignore_eos=True))
        e.step()
        assert s.num_cached_tokens==64
        return s
    def finish():
        e.step()
        assert e.scheduler.is_finished() and not e.states.owners
        e.scheduler.completed.clear();e.scheduler._known_ids.clear()
    report={'status':'running','workload':{'prompt':ids,'past':64,'chunk':64,'batch':1},'profile':{}}
    targets=['before'] if args.baseline else ['after']
    for name in targets:
        fn=variants[name]
        for _ in range(2):begin(fn);finish()
        begin(fn)
        mlp={id(getattr(p.mlp,n)) for p in e.parameters.model.layers for n in ('gate_proj','up_proj','down_proj')}
        proj=lc.project
        def labeled_project(x,node):
            if id(node) in mlp:
                with record_function('MLP_projection'):return proj(x,node)
            return proj(x,node)
        with ExitStack() as stack:
            for obj,attr,label in [(e.scheduler,'schedule','schedule'),(e.runner.builder,'prepare','prepare'),
                    (e.backend,'forward','forward'),(e.backend,'compute_logits','LM_head'),
                    (e.scheduler,'postprocess','postprocess'),(e.states,'synchronize','state_sync'),
                    (lc,'gated_delta_net','GDN'),(lc,'full_attention','FullAttention'),
                    (hf,'torch_chunk_gated_delta_rule','GDN_chunk_core')]:
                stack.enter_context(patch.object(obj,attr,wrap(label,getattr(obj,attr))))
            stack.enter_context(patch.object(lc,'project',labeled_project))
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                with record_function('history_chunk_step'):finish()
        prof.export_chrome_trace(str(out/(name+'.trace.json')))
        report['profile'][name]=[{'name':v.key,'count':v.count,'cpu_ms':v.cpu_time_total/1000,
                                  'cuda_ms':v.device_time_total/1000,'device_type':str(v.device_type)}
                                 for v in prof.key_averages() if v.key in ['history_chunk_step','schedule','prepare','forward','LM_head','postprocess','state_sync','GDN','FullAttention','GDN_chunk_core','MLP_projection']]
    if not args.baseline:
        def capture(fn):
            e.backend.forward=fn;e.runner.capture_logits=True
            s=e.add_request(ids,SamplingParams(max_tokens=1,ignore_eos=True));e.step()
            slot=e.states.lookup(s.seq_id);table=list(s.block_table)
            snaps=[(e.states.conv_pool[:,slot].cpu().clone(),e.states.recurrent_pool[:,slot].cpu().clone())]
            finish()
            logical=torch.arange(128)
            addresses=(torch.tensor(table)[logical//16]*16+logical%16).cuda()
            kv=e.kv_pool.flatten(2,3).index_select(2,addresses).cpu()
            snaps.extend([e.runner.last_logits,kv,e.states.conv_pool[:,slot].cpu().clone(),e.states.recurrent_pool[:,slot].cpu().clone()])
            e.runner.capture_logits=False
            return snaps
        a,b=capture(before),capture(original)
        checks=[torch.equal(x,y) for x,y in zip(a[0],b[0])]+[torch.equal(x,y) for x,y in zip(a[1:],b[1:])]
        assert all(checks)
        report['exact_checks']=dict(zip(['first_conv','first_recurrent','logits','all_valid_KV','final_conv','final_recurrent'],checks))
        report['timings']=[]
        for fn in [before,original]:
            for _ in range(3):begin(fn);finish()
        for pair in range(20):
            row={}
            for name in (['before','after'] if pair%2==0 else ['after','before']):
                begin(variants[name]);torch.cuda.synchronize()
                t=time.perf_counter();finish();row[name]=(time.perf_counter()-t)*1000
            report['timings'].append(row)
        report['median_ms']={name:statistics.median(r[name] for r in report['timings']) for name in variants}
        report['pair_improvement_percent']=[100*(r['before']-r['after'])/r['before'] for r in report['timings']]
    report['status']='complete'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',action='store_true')
    main(p.parse_args())
