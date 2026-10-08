"""Bounded single-GPU diagnostic. No scheduler, pool migration or performance claims."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding, torch_recurrent_gated_delta_rule

from .weight_loader import load_text_parameters
from .layer_compute import decoder_layer,rms_norm


def metric(actual, expected):
    a,b=actual.float(),expected.float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'max_abs':(a-b).abs().max().item(),'rms':(a-b).square().mean().sqrt().item(),
            'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            'finite':finite,'diagnostic_close':finite and torch.allclose(a,b,atol=0.05,rtol=0.01)}


@torch.inference_mode()
def run(model_dir, output, steps=2, recurrent_reference=False):
    if not 0 <= steps <= 4: raise ValueError('bounded diagnostic: decode steps 0..4')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    params,loading=load_text_parameters(model_dir)
    c=params.config['text_config']
    config=Qwen3_5TextConfig(**c)
    config._attn_implementation='eager'
    with torch.device('meta'):
        reference=Qwen3_5ForCausalLM(config)
    reference.load_state_dict(params.state_dict(),strict=True,assign=True)
    reference.tie_weights()
    reference.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config,device='cpu')
    reference=reference.eval().to('cuda:0')
    if recurrent_reference:
        for layer in reference.model.layers:
            if hasattr(layer,'linear_attn'):
                layer.linear_attn.chunk_gated_delta_rule=torch_recurrent_gated_delta_rule
    params=params.eval().to('cuda:0')
    params.tie_weights()
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    rendered=tokenizer.apply_chat_template([{'role':'user','content':'请用一句话说明KV缓存的作用。'}],
                                        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt=tokenizer(rendered,add_special_tokens=False)['input_ids']
    if len(prompt)>64: raise ValueError('diagnostic prompt exceeds 64 tokens')
    cache=DynamicCache(config=config)
    native_states=[None]*len(params.model.layers)
    local_states=[None]*len(params.model.layers)
    captured={}
    def hook(i):
        def record(module,args,kwargs,out):
            incoming=args[0] if args else kwargs['hidden_states']
            captured[i]=(incoming.detach().clone(),out.detach().clone())
        return record
    hooks=[layer.register_forward_hook(hook(i),with_kwargs=True) for i,layer in enumerate(reference.model.layers)]
    results=[]; generated=[]; processed=0
    ids=torch.tensor([prompt],device='cuda:0')
    try:
        for step in range(steps+1):
            positions=torch.arange(processed,processed+ids.shape[1],device='cuda:0')[None]
            expected=reference(input_ids=ids,position_ids=positions,past_key_values=cache,
                               use_cache=True,logits_to_keep=1).logits
            x=F.embedding(ids,params.model.embed_tokens.weight)
            rows=[]
            for i,node in enumerate(params.model.layers):
                ref_input,ref_output=captured[i]
                x,native_states[i]=decoder_layer(x,positions,node,c,i,native_states[i],state_dtype=torch.bfloat16)
                local,local_states[i]=decoder_layer(ref_input,positions,node,c,i,local_states[i],state_dtype=torch.bfloat16)
                row={'layer':i,'type':c['layer_types'][i],
                     'accumulated':metric(x,ref_output),'same_input':metric(local,ref_output)}
                state=native_states[i]; refstate=cache.layers[i]
                if c['layer_types'][i]=='linear_attention':
                    row['conv']=metric(state.conv,refstate.conv_states)
                    row['recurrent']=metric(state.recurrent,refstate.recurrent_states)
                    row['state_dtype']=str(refstate.recurrent_states.dtype)
                else:
                    row['key']=metric(state.key,refstate.keys)
                    row['value']=metric(state.value,refstate.values)
                rows.append(row)
            hidden=rms_norm(x,params.model.norm.weight,c['rms_norm_eps'])
            logits=F.linear(hidden[:,-1:],params.lm_head.weight)
            next_id=expected[:,-1].argmax(-1).item()
            item={'stage':'prefill' if step==0 else f'decode_{step}',
                  'input_tokens':ids.shape[1],'layers':rows,'logits':metric(logits,expected),
                  'reference_token':next_id,'native_token':logits[:,-1].argmax(-1).item(),
                  'first_local_flag':next((r['layer'] for r in rows if not r['same_input']['diagnostic_close']),None)}
            results.append(item)
            print(json.dumps({k:v for k,v in item.items() if k!='layers'}),flush=True)
            generated.append(next_id); processed+=ids.shape[1]
            ids=torch.tensor([[next_id]],device='cuda:0') # teacher-forced same inputs, no branch drift
            captured.clear()
        torch.cuda.synchronize()
        report={'scope':'diagnostic only; no M1 tolerance approval or production GPU backend',
                'reference_prefill':'recurrent_ablation' if recurrent_reference else 'original_chunk',
                'device':torch.cuda.get_device_name(0),'loading':loading,'prompt_ids':prompt,
                'dtype_policy':'checkpoint parameter dtypes; BF16 recurrent storage; FP32 recurrence arithmetic',
                'local_note':'same layer input; own cache history can diverge across decode steps',
                'results':results,'reference_text':tokenizer.decode(generated),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        for handle in hooks: handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--decode-steps',type=int,default=2)
    p.add_argument('--recurrent-reference',action='store_true')
    a=p.parse_args();run(a.model_dir,a.output,a.decode_steps,a.recurrent_reference)
