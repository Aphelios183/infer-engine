"""Controlled, unprofiled scheduling experiment. Production scheduler unchanged.
Teacher-forced append fixes subsequent decode inputs; real forward/head/argmax run.
Not a text-quality benchmark, server load test, or production fairness policy.
"""
import argparse,json,time,statistics,random
from pathlib import Path
from unittest.mock import patch
import torch
from nanovllm import SamplingParams
from qwen35_adapter.gpu_engine import Qwen35Engine

def quantile(xs,p):
    ys=sorted(xs);t=(len(ys)-1)*p;i=int(t);j=min(i+1,len(ys)-1)
    return ys[i]*(j-t)+ys[j]*(t-i) if i!=j else ys[i]

@torch.inference_mode()
def main(output):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    e=Qwen35Engine('/home/ubuntu/huggingface/Qwen3.5-4B',max_model_len=640,max_num_seqs=2,block_size=16)
    base=e.tokenizer('Explain the purpose of a key value cache.',add_special_tokens=False)['input_ids']
    def ids(n):return (base*(n//len(base)+1))[:n]
    fixture=ids(48)
    configs=[('alone',None,False),('decode_priority',128,False),
             ('interleave_whole',None,True),('interleave_256',256,True),
             ('interleave_128',128,True),('interleave_64',64,True)]
    def trial(cfg):
        name,cap,fair=cfg
        e.scheduler.prefill_chunk_size=cap
        e.scheduler.token_budget=512
        def add(n,limit):
            s=e.add_request(ids(n),SamplingParams(max_tokens=limit,ignore_eos=True))
            original=s.append_token
            def forced(_chosen):
                original(fixture[s.num_completion_tokens])
            s.append_token=forced
            return s
        torch.cuda.synchronize()
        begin=time.perf_counter()
        a=add(64,48);b=None;arrival=None;first_b=None
        stamps=[];events=[];streak=0
        while not e.scheduler.is_finished():
            if name!='alone' and b is None and len(stamps)==4:
                arrival=time.perf_counter()
                b=add(512,1)
                streak=0 # fixed four decode steps after B arrival before first service
            has_prefill=bool(e.scheduler.waiting) or any(s.num_cached_tokens<s.num_prompt_tokens for s in e.scheduler.running)
            has_decode=any(s.num_cached_tokens>=s.num_prompt_tokens for s in e.scheduler.running)
            force=fair and streak>=4 and has_prefill and has_decode
            start=time.perf_counter()
            if force:
                # Temporarily decline decode reservations, so ORIGINAL schedule selects prefill.
                # No queue removal, ownership bypass, or production source change.
                with patch.object(e.scheduler.admission,'reserve_decode',return_value=False):
                    batch=e.scheduler.schedule()
                assert batch.is_prefill
            else:batch=e.scheduler.schedule()
            labels=['A' if s is a else 'B' for s in batch.sequences]
            result=e.runner.run(batch)
            done=time.perf_counter()
            assert all(s.num_scheduled_tokens==0 for s in batch.sequences)
            sampled=[labels[row] for row in result.sample_rows]
            if 'A' in sampled:stamps.append(done)
            if 'B' in sampled:first_b=done
            events.append(dict(prefill=batch.is_prefill,requests=labels,counts=batch.input_counts,
                               sampled=sampled,forced_prefill=force,start_ms=(start-begin)*1000,
                               end_ms=(done-begin)*1000))
            streak=0 if batch.is_prefill else streak+1
        end=time.perf_counter()
        assert a.completion_token_ids==fixture
        assert b is None or b.completion_token_ids==fixture[:1]
        assert not e.states.owners and not e.scheduler.admission.blocks.used_block_ids
        assert all(not ('A' in z['requests'] and 'B' in z['requests']) for z in events)
        gaps=[(r-l)*1000 for l,r in zip(stamps,stamps[1:])]
        overlap=[z for z in events if z['prefill'] and z['requests']==['B'] and z['end_ms']<(stamps[-1]-begin)*1000]
        if name=='decode_priority':assert not overlap and first_b>stamps[-1]
        if fair:assert len(overlap)==(512//cap if cap else 1)
        row=dict(config=name,chunk=cap,a_ttft_ms=(stamps[0]-begin)*1000,
                 a_itl_ms=gaps,a_mean_itl_ms=statistics.mean(gaps),a_p95_itl_ms=quantile(gaps,.95),
                 a_p99_itl_ms=quantile(gaps,.99),a_max_itl_ms=max(gaps),
                 b_ttft_ms=None if b is None else (first_b-arrival)*1000,
                 b_prefill_rounds=len([z for z in events if z['prefill'] and z['requests']==['B']]),
                 b_overlap_rounds=len(overlap),total_s=end-begin,
                 output_tokens=48+(b is not None),output_tps=(48+(b is not None))/(end-begin),
                 events=events)
        # Bound Python bookkeeping growth between repeated requests.
        e.scheduler.completed.clear();e.scheduler._known_ids.clear()
        return row
    report=dict(description=__doc__,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                workload={'A_prompt':ids(64),'B_prompt':ids(512),'forced_output':fixture,
                          'B_arrival_after_A_tokens':4,'decode_rounds_per_prefill':4,
                          'repeats':5,'warmups_per_config':1},
                runs=[])
    # Every shape/policy warmed once; rotate seeded order across five rounds.
    for cfg in configs:trial(cfg)
    for repeat in range(5):
        order=list(configs);random.Random(1700+repeat).shuffle(order)
        for cfg in order:
            row=trial(cfg);row['repeat']=repeat;report['runs'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('events','a_itl_ms')}),flush=True)
            output.write_text(json.dumps(report,indent=2))
    report['summary']={}
    for cfg in configs:
        rows=[r for r in report['runs'] if r['config']==cfg[0]]
        keys=['a_mean_itl_ms','a_p95_itl_ms','a_p99_itl_ms','a_max_itl_ms','output_tps','total_s']
        if cfg[0]!='alone':keys.append('b_ttft_ms')
        report['summary'][cfg[0]]={k:{'median':statistics.median(r[k] for r in rows),
                                    'min':min(r[k] for r in rows),'max':max(r[k] for r in rows)} for k in keys}
    report['status']='complete'
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)
    e.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    output=Path(a.output)
    if output.exists():raise RuntimeError('Choose a new result path')
    main(output)
