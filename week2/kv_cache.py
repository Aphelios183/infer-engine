"""Week 2：固定容量 KV Cache + 带缓存的 MHA（推理教学版）。

仅支持同一批样本等长推进；没有位置编码、分页、并发调度或真实模型权重加载。
先读 LESSON_01.md，再结合 step_by_step.py 看每一步形状。
"""

import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# 既支持 python week2/step_by_step.py，也支持从仓库根目录导入。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from week1.mha import MultiHeadAttention  # 复用第一周的权重和完整重算基准


class KVCache:
    """预分配 [B,H,capacity,Dh]，用 length 区分有效内容与未写入空间。"""

    def __init__(self, batch_size, num_heads, max_seq_len, head_dim,
                 device="cpu", dtype=torch.float32):
        dims = (batch_size, num_heads, max_seq_len, head_dim)
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError("缓存各维度必须是正整数")
        if not dtype.is_floating_point:
            raise ValueError("教学版缓存要求浮点 dtype")
        self.k = torch.empty(dims, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.length = 0

    @property
    def capacity(self):
        return self.k.size(2)

    @property
    def used_bytes(self):
        b, h, _, d = self.k.shape
        return 2 * b * h * self.length * d * self.k.element_size()

    @property
    def allocated_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    @torch.no_grad()
    def append(self, k_new, v_new):
        # 先检查，再写入。形状、容量等输入错误不应损坏已有缓存。
        if k_new.ndim != 4 or v_new.shape != k_new.shape:
            raise ValueError("K/V 必须具有相同的 [B,H,new_tokens,Dh] 形状")
        b, h, count, d = k_new.shape
        if (b, h, d) != (self.k.size(0), self.k.size(1), self.k.size(3)) or count <= 0:
            raise ValueError("K/V 的 batch、头数、head_dim 必须匹配，新增长度必须大于零")
        for tensor in (k_new, v_new):
            if tensor.device != self.k.device or tensor.dtype != self.k.dtype:
                raise ValueError("K/V 的 device 和 dtype 必须与缓存一致")
        end = self.length + count
        if end > self.capacity:
            raise ValueError(f"缓存容量不足：需要 {end}，容量 {self.capacity}")
        self.k[:, :, self.length:end, :].copy_(k_new)
        self.v[:, :, self.length:end, :].copy_(v_new)
        self.length = end

    def view(self):
        # 只暴露已写入的前 length 个位置；返回视图，不复制整份缓存。
        return self.k[:, :, :self.length, :], self.v[:, :, :self.length, :]

    def reset(self):
        """开始独立序列时重置有效长度；底层空间保留，旧内容不会被 view 暴露。"""
        self.length = 0


def causal_mask(past_length, new_length, device="cpu"):
    """True 表示屏蔽。新 query 位于绝对位置 past_length ... past+new-1。"""
    if past_length < 0 or new_length <= 0:
        raise ValueError("past_length 必须非负，new_length 必须为正")
    query_positions = torch.arange(past_length, past_length + new_length, device=device)
    key_positions = torch.arange(past_length + new_length, device=device)
    return key_positions[None, :] > query_positions[:, None]


@torch.inference_mode()
def forward_cached(model, x_new, cache, *, explain=False):
    """只输入新增的 hidden states [B,new_tokens,C]；输出也只有新增位置。

    调用方保证 cache 属于同一模型、同一请求、同一批次顺序；换请求先 reset。
    Prefill 就是 cache.length == 0 时一次放入若干 token。
    Decode 常见 new_tokens == 1，也支持多个新增 token 的 chunk。
    """
    if x_new.ndim != 3 or x_new.size(1) <= 0:
        raise ValueError("输入必须是非空的 [B,new_tokens,C]")
    if (x_new.size(0), model.num_heads, model.head_dim) != (
            cache.k.size(0), cache.k.size(1), cache.k.size(3)):
        raise ValueError("模型/输入与缓存维度不匹配")
    if x_new.size(2) != model.d_model:
        raise ValueError("输入 hidden_size 与模型不匹配")
    if x_new.device != cache.k.device or x_new.dtype != cache.k.dtype:
        raise ValueError("输入与缓存的 device/dtype 不匹配")
    if cache.length + x_new.size(1) > cache.capacity:
        raise ValueError("缓存容量不足")
    past_length = cache.length

    # 第一步：仅投影新 token。旧 token 的 K/V 已经存在，无需重新投影。
    q = model._split_heads(model.q_proj(x_new))
    k_new = model._split_heads(model.k_proj(x_new))
    v_new = model._split_heads(model.v_proj(x_new))
    cache.append(k_new, v_new)
    k_all, v_all = cache.view()

    # 第二步：新 query 仍要与全部有效历史 key 做点积。
    scores = q @ k_all.transpose(-2, -1) / math.sqrt(model.head_dim)
    mask = causal_mask(past_length, x_new.size(1), device=x_new.device)
    weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    attended = weights @ v_all
    output = model.out_proj(model._merge_heads(attended))

    if explain:
        print(f"  cache.length: {past_length} -> {cache.length}")
        for name, tensor in [("x_new", x_new), ("Q_new", q), ("K_new", k_new),
                             ("K_all/V_all", k_all), ("scores", scores), ("output", output)]:
            print(f"  {name:14s} {tuple(tensor.shape)}")
        print("  屏蔽矩阵（1=不能看，0=可以看）:")
        print(mask.to(torch.int32).cpu())
        print(f"  有效 K/V: {cache.used_bytes} bytes；预留 K/V: {cache.allocated_bytes} bytes")
    return output
