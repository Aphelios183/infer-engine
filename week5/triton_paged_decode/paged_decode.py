"""教学版单token分页Decode：只读当前层KV池，不接生产引擎。

输入Q已完成需要的Norm/RoPE；当前token的KV已写入池，length包含它。
不处理GDN、写缓存、输出gate/o_proj、Prefill、反向传播或split-K。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode(Q, K, V, Tables, Lengths, Out,
                  H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
                  PAGE: tl.constexpr, TABLE_WIDTH: tl.constexpr,
                  SCALE: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    # 1. 两维launch grid：每个program独占一个[请求,Q头]输出。
    # program不是一个线程；num_warps指定内部协作规模，编译器安排元素。
    request = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (H_Q // H_KV)
    length = tl.load(Lengths + request)  # 标量，包含本轮已写入的token

    # 2. arange创建逻辑索引，不是从内存读取数据。
    # BLOCK_D向上补到2的幂；mask排除超出真实D的列。
    dims = tl.arange(0, BLOCK_D)  # [BLOCK_D]
    q = tl.load(Q + (request * H_Q + q_head) * D + dims,
                mask=dims < D, other=0).to(tl.float32)
    m = tl.full((), -float('inf'), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)

    # 3. length>=1由Python包装器保证；只进入至少有一个有效token的块。
    # 动态循环在GPU执行，不是Python每次提交一个kernel。
    for start in range(0, length, BLOCK_T):
        tokens = start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        valid = tokens < length
        # 连块表加载也要mask：尾部无效token可能对应不存在的逻辑页。
        physical_block = tl.load(Tables + request * TABLE_WIDTH + tokens // PAGE,
                                 mask=valid, other=0).to(tl.int64)
        slots = physical_block * PAGE + tokens % PAGE
        # [:,None]/[None,:]广播成二维元素地址；偏移单位是元素，不是字节。
        offsets = ((slots[:, None] * H_KV + kv_head) * D + dims[None, :])
        mask = valid[:, None] & (dims[None, :] < D)
        k = tl.load(K + offsets, mask=mask, other=0).to(tl.float32)
        v = tl.load(V + offsets, mask=mask, other=0).to(tl.float32)

        # 4. QK沿特征维axis=1归约：[BLOCK_T,BLOCK_D] -> [BLOCK_T]。
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float('inf'))
        # 5. 在线softmax。m/l是标量；acc是BLOCK_D向量。
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        l = alpha * l + tl.sum(p, axis=0)
        # PV沿token维axis=0归约，得到每个输出特征的累积量。
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    # 6. 每个program只有一个写出区域，无跨program归约，不需要atomic_add。
    tl.store(Out + (request * H_Q + q_head) * D + dims, acc / l, mask=dims < D)


@torch.inference_mode()
def paged_decode(q, k_pool, v_pool, block_tables, context_lens, *, block_t=16):
    """q[B,Hq,D], pools[Npages,P,Hkv,D] -> out[B,Hq,D].

    安全优先教学包装器：检查元数据会同步到CPU，不可用于性能宣称/Graph。
    要求同一CUDA设备、连续布局、有限有效Q/K/V、非空历史。
    无效页可以是-1；只有context_lens覆盖的页必须合法。
    """
    tensors=(q,k_pool,v_pool,block_tables,context_lens)
    if any(t.device != q.device for t in tensors) or q.device.type != 'cuda':
        raise ValueError('all tensors must share a CUDA device')
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError('teaching kernel requires contiguous tensors')
    if q.ndim!=3 or k_pool.ndim!=4 or v_pool.shape!=k_pool.shape:
        raise ValueError('expected Q[B,Hq,D] and matching KV[Npage,P,Hkv,D]')
    b,hq,d=q.shape;pages,page,hkv,kd=k_pool.shape
    if min(b,hq,d,pages,page,hkv)<1 or kd!=d or hq%hkv or d>256:
        raise ValueError('invalid head dimensions or GQA ratio')
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or k_pool.dtype!=q.dtype or v_pool.dtype!=q.dtype:
        raise ValueError('matching fp16/bf16/fp32 QKV required')
    if block_tables.ndim!=2 or block_tables.shape[0]!=b or context_lens.shape!=(b,):
        raise ValueError('invalid metadata shape')
    if block_tables.dtype!=torch.int32 or context_lens.dtype!=torch.int32:
        raise ValueError('int32 metadata required')
    if type(block_t) is not int or block_t not in (8,16,32):
        raise ValueError('block_t must be 8,16,32')
    lengths=context_lens.cpu().tolist();tables=block_tables.cpu().tolist()
    for row,n in zip(tables,lengths):
        if not 1<=n<=len(row)*page:
            raise ValueError('nonempty context must fit block table')
        if any(not 0<=bid<pages for bid in row[:triton.cdiv(n,page)]):
            raise ValueError('invalid physical page in effective context')
    out=torch.empty_like(q)
    # 在输入设备的current stream上启动；调用者须保证KV写入已按依赖完成。
    with torch.cuda.device(q.device):
        _paged_decode[(b,hq)](q,k_pool,v_pool,block_tables,context_lens,out,
            H_Q=hq,H_KV=hkv,D=d,PAGE=page,TABLE_WIDTH=block_tables.shape[1],
            SCALE=d**-0.5,BLOCK_T=block_t,BLOCK_D=triton.next_power_of_2(d),num_warps=4)
    return out
