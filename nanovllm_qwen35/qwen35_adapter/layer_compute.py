"""CPU/CUDA eager text-layer math; not a production GPU scheduling backend.

Uses the registered parameter tree without changing checkpoint layout/dtypes.
Caches are explicit per-request values, returned only after successful computation.
No Transformers kernels are called here. GDN prefill is a slow sequential recurrence.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def cpu_float(*tensors):
    # Historical helper name. All operands must now share one CPU/CUDA device.
    for x in tensors:
        if x.device.type not in ('cpu','cuda') or x.device != tensors[0].device or not x.is_floating_point():
            raise ValueError('same-device CPU/CUDA floating tensors required')


def rms_norm(x, weight, eps):
    cpu_float(x, weight)
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * (1 + weight.float())).to(x.dtype)


def gated_rms_norm(x, weight, gate, eps):
    cpu_float(x, weight, gate)
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    y = weight * y.to(x.dtype)
    return (y * F.silu(gate.float())).to(x.dtype)


def split_q_gate(projected, heads, head_dim):
    if projected.shape[-1] != heads * head_dim * 2:
        raise ValueError('Q/gate projection width mismatch')
    return projected.reshape(*projected.shape[:-1], heads, 2 * head_dim).chunk(2, -1)


def partial_rope(q, k, positions, *, theta, rotary_dim):
    """q/k [B,H,T,D], positions [B,T]; text-only default RoPE."""
    cpu_float(q, k)
    if (q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:] or positions.shape != (q.shape[0], q.shape[2])
            or positions.device != q.device or positions.dtype != torch.int64
            or (positions < 0).any()):
        raise ValueError('invalid text positions or Q/K layout')
    if type(rotary_dim) is not int or not 0 < rotary_dim <= q.shape[-1] or rotary_dim % 2:
        raise ValueError('rotary_dim must be positive, even and within head_dim')
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError('invalid rope theta')
    freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim))
    angle = positions.float().unsqueeze(-1) * freq
    angle = torch.cat((angle, angle), -1).unsqueeze(1)
    cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
    def apply(x):
        r, tail = x[..., :rotary_dim], x[..., rotary_dim:]
        a, b = r.chunk(2, -1)
        return torch.cat((r*cos + torch.cat((-b,a), -1)*sin, tail), -1)
    return apply(q), apply(k)


@dataclass(frozen=True)
class FullState:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class GDNState:
    conv: torch.Tensor
    recurrent: torch.Tensor
    length: int


def validate_hidden(x, config):
    cpu_float(x)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[2] != config['hidden_size']:
        raise ValueError('first layer backend supports single request [1,T,hidden] only')


def project(x, node):
    cpu_float(x, node.weight)
    return F.linear(x, node.weight)


@torch.inference_mode()
def full_attention(x, positions, params, config, state=None):
    validate_hidden(x, config)
    h, kvh, d = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    if h % kvh:
        raise ValueError('Q heads must be divisible by KV heads')
    past = 0 if state is None else state.key.shape[2]
    expected_pos = torch.arange(past, past+x.shape[1], dtype=torch.int64, device=x.device).unsqueeze(0)
    if positions.device != x.device or positions.dtype != torch.int64 or not torch.equal(positions, expected_pos):
        raise ValueError('positions must follow cached prefix')
    q, gate = split_q_gate(project(x, params.q_proj), h, d)
    q = rms_norm(q, params.q_norm.weight, config['rms_norm_eps']).transpose(1,2)
    k = project(x, params.k_proj).reshape(1,-1,kvh,d)
    k = rms_norm(k, params.k_norm.weight, config['rms_norm_eps']).transpose(1,2)
    v = project(x, params.v_proj).reshape(1,-1,kvh,d).transpose(1,2)
    rope = config['rope_parameters']
    if rope['rope_type'] != 'default':
        raise ValueError('only default text RoPE supported')
    q, k = partial_rope(q,k,positions,theta=rope['rope_theta'],
                        rotary_dim=int(d*rope.get('partial_rotary_factor',1.0)))
    if state is not None:
        cpu_float(state.key,state.value)
        if (state.key.shape != (1,kvh,past,d) or state.value.shape != state.key.shape
                or state.key.dtype != k.dtype or state.value.dtype != v.dtype):
            raise ValueError('invalid Full cache')
        k, v = torch.cat((state.key,k),2), torch.cat((state.value,v),2)
    all_k, all_v = k.repeat_interleave(h//kvh,1), v.repeat_interleave(h//kvh,1)
    scores = (q @ all_k.transpose(-2,-1)) * d**-0.5
    allowed = torch.arange(k.shape[2],device=x.device)[None,:] <= positions[0,:,None]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probs = scores.float().softmax(-1).to(q.dtype)
    out = (probs @ all_v).transpose(1,2).reshape(1,x.shape[1],h*d)
    out = out * gate.reshape_as(out).sigmoid()
    return project(out, params.o_proj), FullState(k,v)


@torch.inference_mode()
def gated_delta_net(x, params, config, state=None, *, state_dtype=torch.float32):
    """Explicit storage dtype policy. No claim of equivalence to all HF backends.

    FP32 is the default native recurrence storage; BF16 may be selected to examine
    the installed reference cache's rounding. Never mutate incoming cache tensors.
    """
    validate_hidden(x, config)
    if state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('state_dtype must be explicit FP32 or BF16')
    nk, nv = config['linear_num_key_heads'], config['linear_num_value_heads']
    dk, dv = config['linear_key_head_dim'], config['linear_value_head_dim']
    if nv % nk:
        raise ValueError('GDN head ratio invalid')
    kd, vd, width = nk*dk, nv*dv, config['linear_conv_kernel_dim']
    channels = 2*kd+vd
    mixed = project(x,params.in_proj_qkv).transpose(1,2)
    if state is None:
        conv = torch.zeros(1,channels,width,dtype=mixed.dtype,device=x.device)
        recurrent = torch.zeros(1,nv,dk,dv,dtype=state_dtype,device=x.device)
        past = 0
    else:
        cpu_float(state.conv,state.recurrent)
        if (state.conv.shape != (1,channels,width) or state.recurrent.shape != (1,nv,dk,dv)
                or state.conv.dtype != mixed.dtype or state.recurrent.dtype != state_dtype
                or type(state.length) is not int or state.length < 1):
            raise ValueError('invalid GDN state shape/dtype/length')
        conv, recurrent, past = state.conv.clone(), state.recurrent.float(), state.length
    window = torch.cat((conv,mixed),-1)
    next_conv = window[...,-width:].clone()
    mixed = F.silu(F.conv1d(window,params.conv1d.weight,groups=channels)[...,-x.shape[1]:]).transpose(1,2)
    q,k,v = mixed.split([kd,kd,vd],-1)
    q,k = q.reshape(1,-1,nk,dk), k.reshape(1,-1,nk,dk)
    v = v.reshape(1,-1,nv,dv)
    raw_q,raw_k=q,k
    q = q * torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
    k = k * torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    q,k = q.repeat_interleave(nv//nk,2).float(), k.repeat_interleave(nv//nk,2).float()
    beta = project(x,params.in_proj_b).sigmoid().float()
    g = -params.A_log.float().exp() * F.softplus(project(x,params.in_proj_a).float()+params.dt_bias)
    recurrent = recurrent.float()
    if config.get('gdn_prefill_backend') == 'hf_torch_chunk' and (state is None or x.shape[1]>1):
        # Explicit reference-kernel bridge, NOT a native optimized implementation.
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        out,recurrent=torch_chunk_gated_delta_rule(
            raw_q.repeat_interleave(nv//nk,2),raw_k.repeat_interleave(nv//nk,2),v,
            g=g,beta=beta,initial_state=None if state is None else state.recurrent,
            output_final_state=True,use_qk_l2norm_in_kernel=True)
    else:
        output = []
        for i in range(x.shape[1]):
            ki = k[:,i]
            recurrent = recurrent * g[:,i].exp()[...,None,None]
            predicted = (recurrent * ki[...,None]).sum(-2)
            delta = (v[:,i].float()-predicted)*beta[:,i,...,None]
            recurrent = recurrent + ki[...,None]*delta[...,None,:]
            output.append((recurrent*(q[:,i]*dk**-0.5)[...,None]).sum(-2))
        out = torch.stack(output,1).to(x.dtype)
    z = project(x,params.in_proj_z).reshape_as(out)
    out = gated_rms_norm(out,params.norm.weight,z,config['rms_norm_eps'])
    out = project(out.reshape(1,x.shape[1],vd),params.out_proj)
    return out, GDNState(next_conv,recurrent.to(state_dtype),past+x.shape[1])


@torch.inference_mode()
def decoder_layer(x, positions, params, config, layer_idx, state=None, *, state_dtype=torch.float32):
    """Real residual + mixer + MLP calculation over one registered layer node."""
    y = rms_norm(x,params.input_layernorm.weight,config['rms_norm_eps'])
    if config['layer_types'][layer_idx] == 'full_attention':
        y,new_state = full_attention(y,positions,params.self_attn,config,state)
    elif config['layer_types'][layer_idx] == 'linear_attention':
        start = 0 if state is None else state.length
        if not torch.equal(positions,torch.arange(start,start+x.shape[1],device=x.device).unsqueeze(0)):
            raise ValueError('GDN positions must follow cached prefix')
        y,new_state = gated_delta_net(y,params.linear_attn,config,state,state_dtype=state_dtype)
    else:
        raise ValueError('unknown layer type')
    x = x + y
    y = rms_norm(x,params.post_attention_layernorm.weight,config['rms_norm_eps'])
    y = F.silu(project(y,params.mlp.gate_proj))*project(y,params.mlp.up_proj)
    return x+project(y,params.mlp.down_proj), new_state
