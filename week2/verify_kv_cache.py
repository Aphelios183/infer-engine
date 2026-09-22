"""完整前缀作基准，验证单 token / 多 token 增量、边界和缓存生命周期。

python week2/verify_kv_cache.py [--device cuda]
"""
import argparse
import unittest

import torch

try:
    from .kv_cache import KVCache, MultiHeadAttention, causal_mask, forward_cached
except ImportError:
    from kv_cache import KVCache, MultiHeadAttention, causal_mask, forward_cached

DEVICE = "cpu"


class CacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @torch.inference_mode()
    def test_every_chunk_matches_full_recompute(self):
        cases = [(1, 1, [1]), (2, 5, [3, 1, 1]), (2, 7, [2, 3, 2]), (3, 4, [1, 1, 1, 1])]
        for batch, total, chunks in cases:
            with self.subTest(batch=batch, chunks=chunks):
                model = MultiHeadAttention(16, 4).to(DEVICE).eval()
                x = torch.randn(batch, total, 16, device=DEVICE)
                cache = KVCache(batch, 4, total, 4, device=DEVICE)
                start = 0
                for size in chunks:
                    end = start + size
                    expected = model(x[:, :end], causal=True)[:, start:end]
                    actual = forward_cached(model, x[:, start:end], cache)
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                    self.assertEqual(cache.length, end)
                    start = end

    def test_rectangular_mask_uses_absolute_positions(self):
        self.assertEqual(causal_mask(3, 1).tolist(), [[False, False, False, False]])
        self.assertEqual(causal_mask(2, 2).tolist(), [[False, False, False, True], [False] * 4])
        self.assertEqual(causal_mask(0, 1).tolist(), [[False]])

    def test_storage_is_reused_and_bytes_are_counted(self):
        cache = KVCache(2, 4, 5, 3, device=DEVICE)
        ptr = cache.k.data_ptr()
        k = torch.randn(2, 4, 2, 3, device=DEVICE)
        cache.append(k, k + 1)
        self.assertEqual(cache.used_bytes, 2 * 2 * 4 * 2 * 3 * 4)
        self.assertEqual(cache.allocated_bytes, 2 * 2 * 4 * 5 * 3 * 4)
        cache.append(k, k + 1)
        self.assertEqual(cache.k.data_ptr(), ptr)
        torch.testing.assert_close(cache.view()[0][:, :, :2], k)

    def test_bad_append_preserves_existing_contents(self):
        cache = KVCache(1, 2, 3, 4, device=DEVICE)
        k = torch.randn(1, 2, 2, 4, device=DEVICE)
        cache.append(k, k + 1)
        before = [t.clone() for t in cache.view()]
        bad_inputs = [(k, k), (k[:, :, :1].double(), k[:, :, :1].double()),
                      (k[:, :1, :1], k[:, :1, :1]), (k[:, :, :0], k[:, :, :0]),
                      (k[:, :, :1], k)]
        for keys, values in bad_inputs:
            with self.assertRaises(ValueError):
                cache.append(keys, values)
            self.assertEqual(cache.length, 2)
            for old, current in zip(before, cache.view()):
                torch.testing.assert_close(old, current)

    @torch.inference_mode()
    def test_reset_isolates_independent_sequence(self):
        model = MultiHeadAttention(8, 2).to(DEVICE).eval()
        cache = KVCache(1, 2, 4, 4, device=DEVICE)
        forward_cached(model, torch.randn(1, 3, 8, device=DEVICE), cache)
        ptr = cache.k.data_ptr()
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.view()[0].size(2), 0)
        self.assertEqual(cache.k.data_ptr(), ptr)
        x = torch.randn(1, 2, 8, device=DEVICE)
        torch.testing.assert_close(forward_cached(model, x, cache), model(x), atol=1e-5, rtol=1e-4)

    def test_invalid_capacity_rejected(self):
        for capacity in [0, -1]:
            with self.assertRaises(ValueError):
                KVCache(1, 2, capacity, 4, device=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用；可先使用默认 CPU 验证。")
    unittest.main(argv=[__file__, *remaining], verbosity=2)
