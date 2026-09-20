"""第 1 周 · 任务 2：验证你手写的 MHA 是否正确

运行方式：
    conda activate nanovllm
    python week1/verify.py

验证思路（这是你以后写 C++/CUDA 版本时也要用的方法）：
    拿两个"可信参照"和你的实现对比，看数值差多少。
    参照 1：F.scaled_dot_product_attention（PyTorch 内置，业界标准实现）
    参照 2：nn.MultiheadAttention（PyTorch 官方模块）

判断标准：
    最大绝对误差 < 1e-5 视为通过（float32 精度下的正常范围）。
"""

import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mha import MultiHeadAttention, attention
except ImportError:
    from week1.mha import MultiHeadAttention, attention


def report(name: str, mine: torch.Tensor, ref: torch.Tensor, tol: float = 1e-5) -> bool:
    """对比两个张量并打印结果。"""
    max_err = (mine - ref).abs().max().item()
    ok = max_err < tol
    flag = "✅ 通过" if ok else "❌ 失败"
    print(f"  {name:<34} 最大误差 = {max_err:.3e}   {flag}")
    return ok


print("=" * 70)
print("验证 1：与 F.scaled_dot_product_attention 对比")
print("=" * 70)

torch.manual_seed(0)
B, T, C, H = 2, 16, 64, 4
Dh = C // H

q = torch.randn(B, H, T, Dh, dtype=torch.float32)
k = torch.randn(B, H, T, Dh, dtype=torch.float32)
v = torch.randn(B, H, T, Dh, dtype=torch.float32)

results = []

with torch.no_grad():
    mine = attention(q, k, v, causal=True)
    # 参照实现：PyTorch 内置的 SDPA，is_causal=True 自动加因果掩码
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    results.append(report("非因果 vs 因果 (causal=True)", mine, ref))

    mine_nc = attention(q, k, v, causal=False)
    ref_nc = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    results.append(report("无 mask (causal=False)", mine_nc, ref_nc))


print()
print("=" * 70)
print("验证 2：完整 MultiHeadAttention 模块 vs nn.MultiheadAttention")
print("=" * 70)

mine_mha = MultiHeadAttention(d_model=C, num_heads=H).eval()
ref_mha = nn.MultiheadAttention(embed_dim=C, num_heads=H, bias=False).eval()

# 把权重拷过去，保证两者面对同样的参数
ref_mha.in_proj_weight.data.copy_(
    torch.cat([mine_mha.q_proj.weight, mine_mha.k_proj.weight, mine_mha.v_proj.weight], dim=0)
)
ref_mha.out_proj.weight.data.copy_(mine_mha.out_proj.weight)

x = torch.randn(B, T, C)

with torch.no_grad():
    out_mine = mine_mha(x, causal=True)
    # nn.MultiheadAttention 的输入形状是 (T, B, C)，需要转置
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    out_ref, _ = ref_mha(
        x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1),
        attn_mask=causal_mask,
        need_weights=False,
    )
    out_ref = out_ref.transpose(0, 1)
    results.append(report("完整模块 (带因果掩码)", out_mine, out_ref, tol=1e-4))


print()
print("=" * 70)
print("验证 3：因果性检查（逻辑正确性，不依赖参照实现）")
print("=" * 70)

# 因果性的定义：修改位置 t 之后的输入，不应影响位置 t 的输出。
# 这是比"数值对拍"更本质的检查——即使参照实现也错了，这个检查依然有效。
with torch.no_grad():
    x2 = x.clone()
    x2[:, T // 2 :, :] += 100.0          # 大幅扰动后半段

    out1 = mine_mha(x, causal=True)
    out2 = mine_mha(x2, causal=True)

    prefix_diff = (out1[:, : T // 2] - out2[:, : T // 2]).abs().max().item()
    suffix_diff = (out1[:, T // 2 :] - out2[:, T // 2 :]).abs().max().item()

print(f"  前半段输出差异 = {prefix_diff:.3e}  （应 ≈ 0，因为扰动在后面）")
print(f"  后半段输出差异 = {suffix_diff:.3e}  （应明显 > 0，因为被扰动了）")
causal_ok = prefix_diff < 1e-5 and suffix_diff > 1e-2
print(f"  因果性 {'✅ 通过' if causal_ok else '❌ 失败'}")
results.append(causal_ok)


print()
print("=" * 70)
print("性能观察：CPU vs GPU（理解 kernel launch 开销）")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  当前 torch.cuda.is_available() = {torch.cuda.is_available()}")
if device == "cuda":
    print(f"  显卡: {torch.cuda.get_device_name(0)}")


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


if device == "cuda":
    mha_gpu = MultiHeadAttention(C, H).cuda().eval()
    print()
    print(f"  {'seq_len':>8} {'batch':>6} {'耗时(ms)':>12}")
    for seq_len in (1, 16, 64, 256, 1024):
        xs = torch.randn(8, seq_len, C, device="cuda")
        with torch.no_grad():
            ms = timeit(lambda: mha_gpu(xs, causal=True))
        print(f"  {seq_len:>8} {8:>6} {ms:>12.4f}")
    print()
    print("  观察点：seq_len=1 时耗时可能和 seq_len=64 差不多。")
    print("  因为 GPU 上瓶颈是 kernel launch 开销，不是计算量——")
    print("  这正是推理引擎要做 kernel fusion 和 CUDA graph 的原因。")
else:
    print()
    print("  这个终端看不到 GPU，跳过 GPU 计时。")
    print("  在你的终端用空闲显卡跑：")
    print("    CUDA_VISIBLE_DEVICES=2 python week1/verify.py")


print()
print("=" * 70)
if all(results):
    print("全部通过 ✅  你的 MHA 实现是正确的")
    print()
    print("下一步：")
    print("  1. 把这份代码 git add + commit")
    print("  2. 在 verify.py 里加一个你自己想到的边界用例（比如 T=1）")
    print("  3. 准备进入第 2 周：KV Cache")
else:
    print("有检查未通过 ❌  先排查再往下走")
    sys.exit(1)
print("=" * 70)
