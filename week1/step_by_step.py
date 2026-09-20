"""第 1 周 · 任务 1：从零手写 Multi-Head Attention（逐步骤版）

运行方式：
    conda activate nanovllm
    python week1/step_by_step.py

这个文件不追求性能，只做一件事：把 MHA 的每一步 shape 变化打印出来。
你照着屏幕走一遍，形状就再也不会记混。

维度约定（后面所有代码都用这一套）：
    B  = batch size        一批几个样本
    T  = sequence length   每个样本多少个 token
    C  = d_model           模型隐藏维度，也是 Q/K/V 拼接后的总维度
    H  = num_heads         注意力头数
    Dh = head_dim          每个头的维度，Dh = C / H
"""

import math

import torch
import torch.nn.functional as F


def show(name: str, t: torch.Tensor) -> None:
    """打印张量名字和形状，让每一步都可见。"""
    print(f"  {name:<28} {tuple(t.shape)}")


# ---------------------------------------------------------------------------
# 第 0 步：造一份小数据
# ---------------------------------------------------------------------------
print("=" * 70)
print("第 0 步：准备输入")
print("=" * 70)

torch.manual_seed(42)

B, T, C, H = 2, 4, 8, 2          # 故意用很小的数，方便肉眼检查
Dh = C // H                       # = 4

x = torch.randn(B, T, C)          # 输入：一批 token 的隐藏状态

show("x  输入", x)
print(f"  B={B} T={T} C={C} H={H} Dh={Dh}   (C 必须能被 H 整除)")


# ---------------------------------------------------------------------------
# 第 1 步：最朴素的单头注意力
# ---------------------------------------------------------------------------
# Q 是"我在找什么"，K 是"我有什么"，V 是"我实际提供什么"。
# scores[i][j] = 第 i 个 token 对第 j 个 token 的关注程度。

print()
print("=" * 70)
print("第 1 步：单头注意力（先不缩放、不加 mask）")
print("=" * 70)

# 乘 1/sqrt(C) 模拟标准初始化，让 q/k 每个分量的方差 ≈ 1。
# 不这样做的话，方差会随 C 放大，第 2 步的示范数字就失去意义了。
_scale = 1.0 / math.sqrt(C)
W_q = torch.randn(C, C) * _scale
W_k = torch.randn(C, C) * _scale
W_v = torch.randn(C, C) * _scale

q = x @ W_q          # (B, T, C)
k = x @ W_k
v = x @ W_v

# 缩放因子要除以 sqrt(dk)，其中 dk = "点积实际发生的那个维度"。
# 单头情况下 q 的最后一维就是 C，所以 dk = C。
# 这里很容易搞错：dk 不等于 head_dim，除非你已经把头切开了（见第 4 步）。
dk = C

show("q = x @ W_q", q)
show("k = x @ W_k", k)
show("v = x @ W_v", v)

# k 的最后两维转置后才能做矩阵乘： (B,T,C) x (B,C,T) -> (B,T,T)
scores = q @ k.transpose(-2, -1)
show("scores = q @ k^T", scores)
print("    ↑ 每个样本得到一个 T×T 矩阵：第 i 行 = 第 i 个 token 对所有人的打分")


# ---------------------------------------------------------------------------
# 第 2 步：缩放
# ---------------------------------------------------------------------------
# 为什么除以 sqrt(Dh)：
#   q·k 是 Dh 个乘积之和，若 q、k 各分量方差为 1，点积方差 ≈ Dh。
#   Dh 越大，scores 的数值越极端，softmax 会饱和成 one-hot，
#   梯度趋于 0，训练就停了。除以 sqrt(Dh) 把方差拉回 1。

print()
print("=" * 70)
print("第 2 步：缩放（防止 softmax 饱和）")
print("=" * 70)

# 缩放后标准差应该是多少？用公式推，不要凭感觉说"应该 ≈ 1"。
#   scores = Σ(i=1..dk) q_i · k_i
#   var(scores) = dk · var(q) · var(k)        （各分量独立、零均值时）
#   std(scores) = sqrt(dk) · sqrt(var(q)·var(k))
#   除以 sqrt(dk) 后 →  std = sqrt(var(q)·var(k))
# q,k 恰好单位方差时该值为 1。这才是"除以 sqrt(dk)"的准确含义。
scores_scaled = scores / math.sqrt(dk)

print(f"  小样本 (B={B}, T={T})，只有 {scores.numel()} 个 scores：")
print(f"    缩放前 std = {scores.std().item():.4f}，缩放后 std = {scores_scaled.std().item():.4f}")
print(f"    理论值 sqrt(dk) = {math.sqrt(dk):.4f}")
print("    ↑ 偏离理论值很多，因为样本量太小，样本标准差本身噪声极大")
print()

# 用同样的权重投影更多 token，让统计量收敛
with torch.no_grad():
    B_big, T_big = 16, 256
    x_big = torch.randn(B_big, T_big, C)
    q_big = x_big @ W_q
    k_big = x_big @ W_k
    scores_big = q_big @ k_big.transpose(-2, -1)

print(f"  大样本 (B={B_big}, T={T_big})，共 {scores_big.numel():,} 个 scores：")
print(f"    缩放前 std = {scores_big.std().item():.4f}   理论值 sqrt(dk) = {math.sqrt(dk):.4f}")
print(f"    缩放后 std = {(scores_big / math.sqrt(dk)).std().item():.4f}   理论值 = {math.sqrt(q_big.var().item() * k_big.var().item()):.4f}")
print()
print("  结论：点积把 dk 个乘积加起来，方差随 dk 线性增长；")
print("  除以 sqrt(dk) 恰好消掉 dk，把标准差拉回 O(1)，softmax 才不会饱和。")
print()
print("  顺带一个教训：小样本的统计量会骗人。上面 32 个样本给出的 std 是 0.79，")
print("  看起来接近 1 就以为对了；换成 100 万个样本才看清真实规律。")
print()
print("  ⚠️ 这里曾经写错成 sqrt(Dh)：当时 dk 明明是 C=8，却用了 head_dim=4。")
print(f"     错误版本缩放后 std = {(scores / math.sqrt(Dh)).std().item():.4f}，")
print(f"     正确版本缩放后 std = {scores_scaled.std().item():.4f}。")
print("     错误证据其实就打印在屏幕上，但旁边的注释却写着\"≈1\"。")
print("     教训：让数字说话，不要相信注释。")


# ---------------------------------------------------------------------------
# 第 3 步：softmax + causal mask
# ---------------------------------------------------------------------------
# softmax 沿最后一维（dim=-1）做，即"对每个 query 的所有 key 归一化"。
# causal mask：第 i 个 token 只能看 j <= i，未来的 token 必须遮住。
# 遮罩要在 softmax **之前**，且填 -inf 而不是 0——
# 因为 exp(0)=1 会给未来 token 分配真实权重。

print()
print("=" * 70)
print("第 3 步：softmax 与 causal mask")
print("=" * 70)

attn_no_mask = F.softmax(scores_scaled, dim=-1)
show("attn（无 mask）", attn_no_mask)
print("  attn[0][0] 第一行（每个 query 对所有 key 的权重，和应为 1）：")
print(f"    {attn_no_mask[0][0].tolist()}")
print(f"    行和 = {attn_no_mask[0][0].sum().item():.6f}")

# 上三角（不含对角线）置为 -inf
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
show("mask（True=遮住）", mask)
print(f"    mask[0] = {mask[0].tolist()}")

scores_masked = scores_scaled.masked_fill(mask, float("-inf"))
attn = F.softmax(scores_masked, dim=-1)
show("attn（有 mask）", attn)
print("  有 mask 后，第一行只有第 0 个位置有权重：")
print(f"    {attn[0][0].tolist()}")
print(f"    第 0 行：{attn[0][0].tolist()}  ← T=0 只能看自己")
print(f"    第 3 行：{attn[0][3].tolist()}  ← T=3 能看到全部 4 个")


# ---------------------------------------------------------------------------
# 第 4 步：多头 —— 最容易写错的一步
# ---------------------------------------------------------------------------
# 多头的本质：把 C 维切成 H 份，每份 Dh 维，各自独立做注意力，
# 最后再拼回去。切分的顺序决定一切。
#
# 从 (B, T, C) 到 (B, H, T, Dh) 需要三步：
#   1. view(B, T, H, Dh)       把 C 拆成 (H, Dh)——此时 H 在 T 后面
#   2. transpose(1, 2)         把 H 换到 T 前面
#   3. 结果就是 (B, H, T, Dh)
#
# 关键：不能用 view(B, H, T, Dh) 一步到位！那会把 token 维和 head 维
# 搞反，得到的是"错误的头"。必须 view -> transpose。

print()
print("=" * 70)
print("第 4 步：多头（切分）")
print("=" * 70)

# 先把单头的 q/k/v 拼成多头（这里复用上面的 q/k/v 当作已经投影好的结果）
q_mh = q.view(B, T, H, Dh).transpose(1, 2)
k_mh = k.view(B, T, H, Dh).transpose(1, 2)
v_mh = v.view(B, T, H, Dh).transpose(1, 2)

show("q (B,T,C)", q)
show("q.view(B,T,H,Dh)", q.view(B, T, H, Dh))
show("q_mh (B,H,T,Dh)", q_mh)

# 注意 transpose 后内存不连续，取第一个头时用 shape 检查
print(f"  q_mh[0, 0] 是第 0 个样本第 0 个头，形状 {tuple(q_mh[0, 0].shape)}")
print("  ↑ 验证：q_mh[0,0,0] 应等于 q[0,0] 的前 Dh 个元素")
print(f"    q_mh[0,0,0] = {q_mh[0, 0, 0].tolist()}")
print(f"    q[0,0][:Dh] = {q[0, 0, :Dh].tolist()}")

# 多头版本的 scores：(B,H,T,Dh) x (B,H,Dh,T) -> (B,H,T,T)
# 注意 dk 在这里变了：切头之后点积发生在 Dh 维上，不再是 C。
# 缩放因子必须跟着换成 sqrt(Dh)，忘了换就会静默地算错。
dk = Dh
print(f"  切头后点积维度 dk 从 C={C} 变为 Dh={dk}，缩放因子随之改变")
scores_mh = q_mh @ k_mh.transpose(-2, -1) / math.sqrt(Dh)
show("scores_mh (B,H,T,T)", scores_mh)
print("  ↑ 每个头都有一张自己独立的 T×T 注意力图")


# ---------------------------------------------------------------------------
# 第 5 步：合并头 + 输出投影
# ---------------------------------------------------------------------------
# 反向操作：(B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, C)
# 注意 transpose 之后内存不连续，必须 .contiguous() 才能 view。
# 这是新手 100% 会踩的坑：不加 contiguous 会报
# "view size is not compatible with input tensor's size and stride"。

print()
print("=" * 70)
print("第 5 步：合并头与输出投影")
print("=" * 70)

mask_mh = mask.unsqueeze(0).unsqueeze(0)          # (1,1,T,T) 广播到 (B,H,T,T)
attn_mh = F.softmax(scores_mh.masked_fill(mask_mh, float("-inf")), dim=-1)
show("attn_mh (B,H,T,T)", attn_mh)

out_mh = attn_mh @ v_mh                            # (B,H,T,Dh)
show("out_mh (B,H,T,Dh)", out_mh)

out = out_mh.transpose(1, 2)                       # (B,T,H,Dh)
show("out.transpose(1,2)", out)

out = out.contiguous().view(B, T, C)               # (B,T,C)  ← 必须 contiguous
show("out (B,T,C)", out)

W_o = torch.randn(C, C) * _scale
out = out @ W_o
show("最终输出 out @ W_o", out)

print()
print("=" * 70)
print("完成。形状走完一圈：(B,T,C) -> ... -> (B,T,C)")
print("下一步：python week1/verify.py  验证数值是否正确")
print("=" * 70)
