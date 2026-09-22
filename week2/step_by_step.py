"""第一课：Prefill 3 个 token，再逐个 Decode 2 个 token。

先猜形状和 mask，再运行；CPU 即可完成本课。
"""
import argparse
import math

import torch

try:
    from .kv_cache import KVCache, MultiHeadAttention, forward_cached
except ImportError:
    from kv_cache import KVCache, MultiHeadAttention, forward_cached


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用；本课可直接使用默认 CPU。")
    torch.manual_seed(42)
    b, total, hidden, heads = 1, 5, 8, 2
    model = MultiHeadAttention(hidden, heads).to(args.device).eval()
    x = torch.randn(b, total, hidden, device=args.device)
    cache = KVCache(b, heads, max_seq_len=8, head_dim=hidden // heads, device=args.device)

    print("本课使用随机 hidden states，不是 tokenizer 或真实语言模型。")
    print("同一组权重、同一段输入，比较完整重算与缓存计算。")
    print("先预测：Prefill 3 个后，第 4 个 token 的 Q、K_all、scores 是什么形状？\n")
    expected = model(x[:, :3], causal=True)
    print("步骤 1 / Prefill：空缓存，一次输入 x1,x2,x3")
    actual = forward_cached(model, x[:, :3], cache, explain=True)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    print(f"  与完整前缀对拍，最大误差 {(actual - expected).abs().max().item():.3e}\n")

    for index in range(3, total):
        print(f"步骤 {index - 1} / Decode：只输入 x{index + 1}")
        expected = model(x[:, :index + 1], causal=True)[:, -1:]
        actual = forward_cached(model, x[:, index:index + 1], cache, explain=True)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        print(f"  与完整前缀最后位置对拍，最大误差 {(actual - expected).abs().max().item():.3e}\n")

    print("步骤 4 / 为什么矩形 mask 容易写错？")
    q = torch.ones(1, 1, 1, 2, device=args.device)
    k = torch.ones(1, 1, 4, 2, device=args.device)
    v = torch.tensor([10., 20., 30., 40.], device=args.device).view(1, 1, 4, 1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(2)
    correct = scores.softmax(-1) @ v  # 第 4 个 query 可看所有 4 个 key
    wrong_mask = torch.triu(torch.ones(1, 4, dtype=torch.bool, device=args.device), diagonal=1)
    wrong = scores.masked_fill(wrong_mask, float("-inf")).softmax(-1) @ v
    print(f"  所有 key 得分一样，正确输出=四个 V 的平均值={correct.item():.1f}")
    print(f"  错把新 query 当作绝对位置 0，输出={wrong.item():.1f}，只看到了第一个 V。")
    assert correct.item() == 25.0 and wrong.item() == 10.0

    print("\n对拍通过。缓存减少历史投影与旧 query 的重复计算，仍然读取历史 K/V。")
    print("下一步：读 LESSON_01.md，完成 EXERCISES.md 的第一课题目。")


if __name__ == "__main__":
    main()
