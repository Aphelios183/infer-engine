"""为什么除以 sqrt(Dh)？softmax 为什么沿 key 维？

运行：
    conda activate nanovllm
    python week1/why_scale_and_key_dim.py

一句话答案：
  1. 除以 sqrt(Dh)：点积会把 Dh 个乘积加起来，数值自然变大（标准差 ≈ sqrt(Dh)）。
     softmax 对数值大小极其敏感，数值一大就变成"只有最大的那个"，其余全变 0，
     梯度随之消失。除以 sqrt(Dh) 把这个增长消掉。
  2. softmax 沿 key 维：每个 query 要产出一个"所有 value 的加权平均"。
     加权平均要求权重和为 1，而这些权重是沿 key 方向分配的，所以归一化也沿 key 维。

下面把这两句话变成可以看见的数字。
"""

import math

import torch
import torch.nn.functional as F


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ===========================================================================
# 第一部分：点积为什么"自然就变大"
# ===========================================================================
banner("第一部分：Dh 越大，点积越大 —— 这跟谁更相关没有关系")

print("""
  Q·K 是 Dh 个乘积之和：
      scores = q1*k1 + q2*k2 + ... + q_Dh*k_Dh

  假设 q、k 的每个分量都是均值 0、方差 1 的随机数，那么：
      单个乘积 q_i*k_i 的方差  = 1
      Dh 个独立项相加，方差相加  = Dh          ← 注意是"方差"相加
      所以 std(scores) = sqrt(Dh)              ← 标准差是 sqrt

  也就是说：Dh 变大，scores 天然就变大，和"这两个向量到底像不像"无关。
  下面用大样本验证这个规律（小样本的统计量会骗人，详见 week1/step_by_step.py 第 2 步）。
""")

print(f"  {'Dh':>6} {'实测 std(scores)':>18} {'理论 sqrt(Dh)':>16}")
torch.manual_seed(0)
for dh in (4, 16, 64, 256, 1024):
    q = torch.randn(20000, dh)
    k = torch.randn(20000, dh)
    scores = (q * k).sum(dim=-1)
    print(f"  {dh:>6} {scores.std().item():>18.3f} {math.sqrt(dh):>16.3f}")
print("  ↑ 完全吻合。Dh=1024 时 scores 的标准差是 32，比 Dh=4 时大了 16 倍。")


# ===========================================================================
# 第二部分：softmax 对数值大小极其敏感
# ===========================================================================
banner("第二部分：softmax 不是等比放大不变的，数值一大就退化")

print("""
  softmax 有个关键性质：softmax(c·x) ≠ softmax(x)。
  把输入整体放大 c 倍，输出会变得更"尖锐"。
""")

base = torch.tensor([2.0, 1.0, 0.5, -1.0])
print(f"  原始打分: {base.tolist()}")
print()
print(f"  {'缩放':>8} {'softmax 输出':>52} {'熵':>8}")
for c in (1.0, 2.0, 4.0, 8.0, 16.0):
    p = F.softmax(base * c, dim=-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum().item()
    vals = "[" + ", ".join(f"{x:.4f}" for x in p.tolist()) + "]"
    print(f"  x{c:>6.0f} {vals:>52} {entropy:>8.4f}")

print("""
  ↑ 缩放倍数越大，权重越集中到第一个位置，熵（不确定度）越低。
    缩放 16 倍时几乎就是 [1, 0, 0, 0]——这个状态叫"softmax 饱和"。

  饱和为什么是灾难？
    梯度里含有 p·(1-p) 这样的因子。看第一个位置：
      未饱和：p=0.66，p(1-p) = 0.22      ← 梯度正常
      已饱和：p≈1.00，p(1-p) ≈ 0.0000x   ← 梯度几乎为 0
    梯度为 0 意味着参数不再更新，训练就此停住。
    所以不是"数值大不好看"，而是"数值大就没法学习"。
""")


# ===========================================================================
# 第三部分：这三件事串起来
# ===========================================================================
banner("第三部分：把前两部分合起来，就得到了 sqrt(Dh)")

print("""
  第一步：点积让 scores 的标准差变成 sqrt(Dh)
  第二步：softmax 对数值大小敏感，数值大就饱和，梯度消失
  结论：  除以 sqrt(Dh)，把标准差从 sqrt(Dh) 拉回 1（前提是 q、k 单位方差）

  一次除法的意义：让 scores 的"尺度"不随 Dh 变化。
  这样无论 head_dim 是 64 还是 128，softmax 都工作在正常区间。
""")


# ===========================================================================
# 第四部分：softmax 为什么沿 key 维
# ===========================================================================
banner("第四部分：softmax 为什么沿 key 维（dim=-1）")

torch.manual_seed(0)
T_q, T_k = 3, 4
scores = torch.randn(T_q, T_k) * 2

print("  scores 的形状是 (T_q, T_k) —— 行是 query，列是 key：")
print()
print(f"  {'':>10}" + "".join(f"{'key'+str(j):>12}" for j in range(T_k)))
for i in range(T_q):
    print(f"  {'query'+str(i):>10}" + "".join(f"{scores[i, j].item():>12.3f}" for j in range(T_k)))

print("""
  语义：第 i 行 = "第 i 个 token（query）给每个 token（key）打的分"

  接下来要回答的问题是：第 i 个 token 应该把多少注意力分给各个 key？
  这是一个"分配"问题——每个 query 手里有 1.0 的注意力预算，
  要按打分高低分给所有 key。所以是【每一行内部】归一化。
""")

row_norm = F.softmax(scores, dim=-1)
col_norm = F.softmax(scores, dim=-2)

print("  正确做法：softmax(dim=-1)，沿 key 维 —— 每一行的和为 1")
for i in range(T_q):
    print(f"    query{i}: {[round(x, 3) for x in row_norm[i].tolist()]}  行和 = {row_norm[i].sum().item():.3f}")

print()
print("  错误做法：softmax(dim=-2)，沿 query 维 —— 每一列的和为 1")
for j in range(T_k):
    print(f"    key{j}:   {[round(x, 3) for x in col_norm[:, j].tolist()]}  列和 = {col_norm[:, j].sum().item():.3f}")

print("""
  为什么列归一化是错的？
    "每个 key 从所有 query 那里收到的权重之和等于 1"——这句话没有物理意义。
    一个 token 完全可以被很多 token 同时关注，它收到的总权重本来就该大于 1。
    只有"每个 query 的预算为 1"这个约束是成立的。

  更严格地说，下一步要做 attn @ V：
      output[i] = Σ(j) attn[i][j] · V[j]
  这是"用 attn[i] 当权重，把所有 V 加权平均"。
  加权平均要求权重非负且和为 1（凸组合），否则输出的数值范围会随 T_k 漂移，
  网络每一层的数值尺度都无法控制。
  而 attn[i] 是沿 key 方向的一组权重 —— 所以必须把 key 方向归一化。
""")


# ===========================================================================
# 第五部分：和 causal mask 的配合
# ===========================================================================
banner("第五部分：加了 causal mask 之后，行和还是 1 吗？")

T2 = 4
scores_sq = torch.randn(T2, T2) * 2
mask = torch.triu(torch.ones(T2, T2, dtype=torch.bool), diagonal=1)
masked = scores_sq.masked_fill(mask, float("-inf"))
p = F.softmax(masked, dim=-1)

print("  遮罩位置填 -inf，softmax 之后恰好等于 exp(-inf)=0，所以：")
for i in range(T2):
    print(f"    query{i}: {[round(x, 3) for x in p[i].tolist()]}  行和 = {p[i].sum().item():.3f}")
print("""
  ↑ 行和依然是 1（被遮的位置贡献 0）。
    这就是"填 -inf 而不是填 0"的原因：
      - 填 -inf：遮住的位置权重精确为 0，剩下的权重自动重新分配，行和为 1
      - 填 0：   遮住的位置会拿到 exp(0)=1 的权重，等于在偷看未来
""")

banner("总结：两条规则，一个目的")
print("""
  除以 sqrt(Dh)   → 控制 scores 的【尺度】，让 softmax 不饱和、梯度能传
  沿 key 维归一化  → 控制权重的【语义】，让它成为合法的凸组合权重

  两条规则都在做同一件事：让数值稳定、含义明确。
  这就是为什么它们一个都不能省。
""")
