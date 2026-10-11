"""Read-only numerical localization; separate profiler run. No production math edits."""
import json, argparse
from pathlib import Path
from unittest.mock import patch
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from nanovllm import SamplingParams
from qwen35_adapter.gpu_engine import Qwen35Engine
from qwen35_adapter import pool_backend as pb, layer_compute as lc
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

def metric(a,b):
    a,b=a.float(),b.float()
    return dict(max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item(),exact=torch.equal(a,b))

def statecopy(s):
    names=('key','value') if isinstance(s,lc.FullState) else ('conv','recurrent')
    return {k:getattr(s,k).detach().cpu().clone() for k in names}

@torch.inference_mode()
def main(out):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    e=Qwen35Engine('/home/ubuntu/huggingface/Qwen3.5-4B',max_model_len=256,max_num_seqs=1,block_size=4)
    base=e.tokenizer('Explain the purpose of a key value cache.',add_special_tokens=False)['input_ids']
    ids=(base*20)[:65]
    report={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'ids':ids}
    original=pb.decoder_layer
    def run(tokens,cap,capture=True):
        records=[]
        def hooked(x,pos,p,c,lid,state=None,**kw):
            incoming=statecopy(state) if capture and state is not None else None
            y,s=original(x,pos,p,c,lid,state,**kw)
            if capture: records.append(dict(layer=lid,pos=pos.cpu().tolist()[0],x=x.cpu().clone(),y=y.cpu().clone(),state=statecopy(s),incoming=incoming))
            return y,s
        e.scheduler.prefill_chunk_size=cap
        e.runner.capture_logits=capture
        e.add_request(tokens,SamplingParams(max_tokens=1,ignore_eos=True))
        with patch.object(pb,'decoder_layer',hooked):
            while not e.scheduler.is_finished():e.step()
        return records,e.runner.last_logits
    full,fl=run(ids,None); seg,sl=run(ids,64); prefix,_=run(ids[:64],None)
    report['logits']=metric(sl,fl)
    report['layers']=[]
    for i in range(32):
        a,b,d=full[i],seg[i],seg[32+i]
        report['layers'].append(dict(layer=i,type=e.backend.config['layer_types'][i],
            first_chunk_vs_prefix=metric(b['y'],prefix[i]['y']),
            first_chunk_vs_full=metric(b['y'],a['y'][:,:64]),
            tail_input=metric(d['x'],a['x'][:,64:]),tail_output=metric(d['y'],a['y'][:,64:]),
            history_readback={k:metric(d['incoming'][k],b['state'][k]) for k in b['state']},
            boundary_vs_prefix={k:metric(b['state'][k],prefix[i]['state'][k]) for k in b['state']},
            final_states={k:metric(d['state'][k],a['state'][k]) for k in d['state']},
            tail_positions=d['pos']))
    print('layer sweep done',flush=True)
    # Hold layer-0 input fixed; capture projection and GDN core inputs independently.
    x=full[0]['x'].cuda(); p=e.parameters.model.layers[0];c=e.backend.config
    norm=lc.rms_norm(x,p.input_layernorm.weight,c['rms_norm_eps'])
    original_core=hf.torch_chunk_gated_delta_rule
    captured=[]
    def core(q,k,v,**kw):
        captured.append((q.clone(),k.clone(),v.clone(),{n:t.clone() if torch.is_tensor(t) else t for n,t in kw.items()}))
        return original_core(q,k,v,**kw)
    with patch.object(hf,'torch_chunk_gated_delta_rule',core):
        yf,sf=lc.gated_delta_net(norm,p.linear_attn,c,state_dtype=torch.bfloat16)
        y1,s1=lc.gated_delta_net(norm[:,:64],p.linear_attn,c,state_dtype=torch.bfloat16)
        y2,s2=lc.gated_delta_net(norm[:,64:],p.linear_attn,c,s1,state_dtype=torch.bfloat16)
    report['isolated_layer0']={'output':metric(torch.cat((y1,y2),1),yf),
        'projection':{name:metric(torch.cat([lc.project(norm[:,:64],getattr(p.linear_attn,name)),lc.project(norm[:,64:],getattr(p.linear_attn,name))],1),lc.project(norm,getattr(p.linear_attn,name))) for name in ['in_proj_qkv','in_proj_a','in_proj_b','in_proj_z']}}
    q,k,v,kw=captured[0]
    def calc(fn,q,k,v,g,beta,state=None):
        return fn(q,k,v,g=g,beta=beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
    g,beta=kw['g'],kw['beta']
    whole,ws=calc(original_core,q,k,v,g,beta)
    first,fs=calc(original_core,q[:,:64],k[:,:64],v[:,:64],g[:,:64],beta[:,:64])
    report['core_same_inputs']={}
    for dtype in [torch.float32,torch.bfloat16]:
        for name,fn in [('chunk',original_core),('recurrent',hf.torch_recurrent_gated_delta_rule)]:
            tail,ts=calc(fn,q[:,64:],k[:,64:],v[:,64:],g[:,64:],beta[:,64:],fs.to(dtype))
            report['core_same_inputs'][str(dtype)+'_'+name]={'output':metric(torch.cat((first,tail),1),whole),'state':metric(ts,ws)}
    # Sequential recurrence over identical operands is a diagnostic alternate evaluation.
    rec,rs=calc(hf.torch_recurrent_gated_delta_rule,q,k,v,g,beta)
    report['core_same_inputs']['whole_recurrent_vs_chunk']={'output':metric(rec,whole),'state':metric(rs,ws)}
    # Diagnostic precision ablations only; restore production policy afterwards.
    report['ablations']={}
    base_project=lc.project
    base_pool=e.states.recurrent_pool
    def fp32_project(t,node):
        return torch.nn.functional.linear(t.float(),node.weight.float()).to(t.dtype)
    for name,proj32,state32 in [('state_fp32',False,True),('projection_fp32',True,False),('both_fp32',True,True)]:
        e.states.recurrent_pool=base_pool.float() if state32 else base_pool
        with patch.object(lc,'project',fp32_project if proj32 else base_project):
            _,a=run(ids,None);_,b=run(ids,64)
        report['ablations'][name]=metric(b,a)
        print(name,report['ablations'][name],flush=True)
    e.states.recurrent_pool=base_pool
    report['layer0_input_norm']=metric(lc.rms_norm(x[:,:64],p.input_layernorm.weight,c['rms_norm_eps']),norm[:,:64])
    report['projection_prefix_tail']={}
    for name in ['in_proj_qkv','in_proj_a','in_proj_b','in_proj_z']:
        node=getattr(p.linear_attn,name)
        w=base_project(norm,node);a=base_project(norm[:,:64],node);b=base_project(norm[:,64:],node)
        report['projection_prefix_tail'][name]={'prefix':metric(a,w[:,:64]),'tail':metric(b,w[:,64:]),
            'fp32':metric(torch.cat((fp32_project(norm[:,:64],node),fp32_project(norm[:,64:],node)),1),fp32_project(norm,node))}
    # Separate profiling: NO tensor snapshots or diagnostic D2H clones.
    run(ids,None,False);run(ids,64,False)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        with record_function('whole_prefill_65'):run(ids,None,False)
        with record_function('chunked_prefill_64_plus_1'):run(ids,64,False)
    prof.export_chrome_trace(str(out/'trace.json'))
    report['profile']=[{'name':z.key,'count':z.count,'cpu_us':z.cpu_time_total,'device_us':z.device_time_total} for z in prof.key_averages() if z.key in ['whole_prefill_65','chunked_prefill_64_plus_1','aten::linear','aten::matmul','aten::conv1d','aten::to','aten::copy_','cudaStreamSynchronize']]
    (out/'report.json').write_text(json.dumps(report,indent=2))
    e.close()
    print(json.dumps({'logits':report['logits'],'first_layer':report['layers'][0],'isolated':report['isolated_layer0'],'core':report['core_same_inputs'],'profile':report['profile']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    main(out)
