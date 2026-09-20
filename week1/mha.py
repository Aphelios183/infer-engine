"""Multi-Head Attention 的可复用实现（第 1 周产出）

这是一个"教学版"实现：正确、清晰、带注释，但不做性能优化。
后面的 C++ / CUDA 版本会以它为正确性基准（对拍）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """生成 causal mask，True 表示该位置被遮住。

    形状 (T, T)，上三角（不含对角线）为 True：
        位置 i 只能看到 j <= i，未来的 j > i 全部遮住。
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def attention(q, k, v, causal: bool = True):
    """缩放点积注意力。

    参数形状：
        q: (B, H, T_q, Dh)
        k: (B, H, T_k, Dh)
        v: (B, H, T_k, Dh)
    返回：
        (B, H, T_q, Dh)

    三件事按顺序做完：
        1. scores = q @ k^T / sqrt(Dh)      —— 打分并缩放
        2. mask 掉未来位置（填 -inf）        —— 必须在 softmax 之前
        3. softmax 归一化后加权求和 v
    """
    dh = q.size(-1)

    # 1. 打分。1/sqrt(dh) 把方差拉回 1，避免 softmax 饱和。
    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)   # (B,H,T_q,T_k)

    # 2. 遮罩。填 -inf 而非 0，因为 exp(0)=1 会给出真实权重。
    if causal:
        t_q, t_k = scores.size(-2), scores.size(-1)
        mask = build_causal_mask(t_k, device=scores.device)[-t_q:, :]
        scores = scores.masked_fill(mask, float("-inf"))

    # 3. 归一化 + 加权求和。dim=-1 表示对 key 维归一化。
    weights = F.softmax(scores, dim=-1)
    return weights @ v


class MultiHeadAttention(nn.Module):
    """标准多头注意力。

    d_model 必须能被 num_heads 整除，head_dim = d_model // num_heads。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, H, T, Dh)

        必须走 view -> transpose 两步。
        直接 view(B, H, T, Dh) 会把 token 维和 head 维搞反，得到错误的头。

        注意：这里用 x 而不是 t 作参数名。如果写成 t，
        下一行的 "b, t, _ = t.shape" 会把 t 从张量重绑定成整数，
        张量引用就丢了——这是很常见的变量遮蔽陷阱。
        """
        b, seq_len, _ = x.shape
        return x.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Dh) -> (B, T, C)

        transpose 之后内存不连续，必须先 contiguous() 才能 view()。
        少了这一步会报 "view size is not compatible ... stride"。
        """
        b, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        out = attention(q, k, v, causal=causal)     # (B,H,T,Dh)

        return self.out_proj(self._merge_heads(out))
