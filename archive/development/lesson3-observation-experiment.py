# Archived exact diagnostic experiment; use PYTHONPATH=. from project root.
# Writes a new report and refuses to overwrite an existing result.
import json, torch
from pathlib import Path
from week3.reference_runtime import load_reference
from week3.minimal_generate import run_greedy
from week3.inspect_model import inspect_prompt
from week3.cache_inspection import snapshot_cache

destination=Path('week3/results/qwen35-cache-observation.json')
if destination.exists():
    raise FileExistsError(destination)
model,tok,manifest,eos=load_reference('/home/ubuntu/huggingface/Qwen3.5-4B','cuda:0')
torch.cuda.synchronize()
load_peak=torch.cuda.max_memory_allocated()
parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
kinds=model.config.text_config.layer_types
runs=[]
for name,prompt in [('A','用一句话解释 KV Cache，并说明它为什么能够避免重复计算历史 token 的 K 和 V。'),('B','你好')]:
    ids=torch.tensor([inspect_prompt(tok,prompt,thinking=False)['chat_ids']],dtype=torch.long,device='cuda')
    previous={}
    def observe(cache,n):
        torch.cuda.synchronize()
        record=snapshot_cache(cache,kinds,n)
        assert record['complete']
        # Snapshot only selected layers; these clones are diagnostic overhead.
        selected={'keys':cache.layers[3].keys,'values':cache.layers[3].values,
                  'conv':cache.layers[0].conv_states,'recurrent':cache.layers[0].recurrent_states}
        changes={}
        for key,t in selected.items():
            if key in previous:
                before=previous[key]
                if key in ('keys','values'):
                    preserved=torch.equal(before,t[...,:before.shape[-2],:])
                    assert preserved, 'history overwritten'
                    changes[key]={'history_exactly_preserved':preserved}
                else:
                    assert before.shape==t.shape
                    changes[key]={'same_shape':True,
                                  'finite':bool(torch.isfinite(t).all().item()),
                                  'max_abs_change':float((t.float()-before.float()).abs().max().item())}
            previous[key]=t.detach().clone()
        record['selected_layer_changes']=changes
        record['diagnostic_clone_bytes']=sum(t.numel()*t.element_size() for t in previous.values())
        record['allocator_with_diagnostics']={
            'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'request_peak_allocated':torch.cuda.max_memory_allocated()}
        return record
    torch.cuda.reset_peak_memory_stats()
    result=run_greedy(model,ids,eos,4,observe_cache=observe)
    assert result['trace'][0]['position_start']==0
    assert result['trace'][0]['cache']['processed_tokens']==ids.shape[1]
    result['request']=name
    result['text']=tok.decode(result['output_ids'],skip_special_tokens=True)
    result['prompt']=prompt
    previous.clear()
    torch.cuda.synchronize()
    result['after_request_allocated']=torch.cuda.memory_allocated()
    result['after_request_reserved']=torch.cuda.memory_reserved()
    runs.append(result)
    print(name,'input',ids.shape[1],'output',result['generated_tokens'],'stop',result['stop_reason'],flush=True)
    for event in result['trace']:
        c=event['cache']
        print(event['phase'],event['processed_tokens'],c['logical_tensor_bytes'],c['unique_storage_bytes'],c['selected_layer_changes'],flush=True)
report={'model':manifest,'gpu_physical_index':5,'gpu_uuid':'GPU-093b8773-f073-4c18-87c7-2afa616fd9c2',
        'dtype':str(model.dtype),'parameter_bytes':parameter_bytes,'load_peak_allocated':load_peak,
        'runs':runs,'scope':'diagnostic with extra clones; not performance or numerical parity acceptance'}
with destination.open('x',encoding='utf-8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print('CACHE_OBSERVATION_COMPLETE',flush=True)
