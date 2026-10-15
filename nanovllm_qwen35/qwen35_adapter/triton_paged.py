"""Experimental read-only paged decode; Runner validates metadata before upload.

Same online-softmax algorithm as week5/triton_paged_decode/paged_decode.py.
No CPU metadata reads here. One program per request/Q head; not split-K.
Caller must provide valid contiguous CUDA pools/tables, nonempty lengths,
finite effective QKV, and complete KV writes on the current stream.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, Tables, Lengths, Out,
            HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            PAGE: tl.constexpr, WIDTH: tl.constexpr,
            BT: tl.constexpr, BD: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    kh = head // (HQ // HK)
    length = tl.load(Lengths + row)
    ds = tl.arange(0, BD)
    q = tl.load(Q + (row * HQ + head) * D + ds, ds < D, other=0).to(tl.float32)
    m = tl.full((), -float('inf'), tl.float32)
    l = tl.full((), 0., tl.float32)
    acc = tl.full((BD,), 0., tl.float32)
    for start in range(0, length, BT):
        ts = start + tl.arange(0, BT)
        valid = ts < length
        page = tl.load(Tables + row * WIDTH + ts // PAGE, valid, other=0).to(tl.int64)
        slots = page * PAGE + ts % PAGE
        offsets = (slots[:, None] * HK + kh) * D + ds[None, :]
        mask = valid[:, None] & (ds[None, :] < D)
        k = tl.load(K + offsets, mask, other=0).to(tl.float32)
        v = tl.load(V + offsets, mask, other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        scores = tl.where(valid, scores, -float('inf'))
        new_m = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m)
        l = alpha * l + tl.sum(p, 0)
        acc = alpha * acc + tl.sum(p[:, None] * v, 0)
        m = new_m
    tl.store(Out + (row * HQ + head) * D + ds, acc / l, ds < D)


@torch.inference_mode()
def paged_decode_trusted(q, k, v, tables, lengths, *, block_t=16):
    """Internal trusted-metadata API: Q[B,H,D], KV[pages,page,Hkv,D].

    Shape/layout checks do not synchronize CUDA. Value checks belong to Runner.
    Do not expose to unvalidated user metadata or use as a Graph-ready claim.
    """
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('invalid Q/K/V shape')
    b, h, d = q.shape
    pages, page, hk, kd = k.shape
    if min(b,h,d,pages,page,hk) < 1 or d != kd or d > 256 or h % hk:
        raise ValueError('invalid head dimensions')
    tensors = (q,k,v,tables,lengths)
    if q.device.type != 'cuda' or any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('contiguous same-device CUDA tensors required')
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('matching floating QKV required')
    if tables.ndim != 2 or tables.shape[0] != b or tables.shape[1] < 1 or lengths.shape != (b,):
        raise ValueError('invalid metadata shape')
    if tables.dtype != torch.int32 or lengths.dtype != torch.int32 or block_t not in (8,16,32):
        raise ValueError('invalid metadata dtype or tile')
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _decode[(b,h)](q,k,v,tables,lengths,out,h,hk,d,page,tables.shape[1],
                       block_t,triton.next_power_of_2(d),num_warps=4)
    return out
