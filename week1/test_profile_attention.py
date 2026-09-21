"""CPU-only correctness/parser tests; GPU measurements are exercised separately."""
import unittest

import torch

from mha import MultiHeadAttention, build_causal_mask
from profile_attention import attention_forward, summarize_trace, union_duration


class ProfileExperimentTest(unittest.TestCase):
    def test_variants_share_weights_and_outputs(self):
        torch.manual_seed(7)
        for batch, length, hidden, heads in [(1, 1, 8, 2), (2, 4, 16, 4), (1, 17, 24, 3)]:
            model = MultiHeadAttention(hidden, heads).eval()
            x = torch.randn(batch, length, hidden)
            with torch.inference_mode():
                expected = model(x)
                for kwargs in [{}, {"cached_mask": build_causal_mask(length)}, {"sdpa": True}, {"annotate": True}]:
                    torch.testing.assert_close(attention_forward(model, x, **kwargs), expected, atol=1e-5, rtol=1e-4)

    def test_interval_union_handles_overlap(self):
        self.assertEqual(union_duration([(5, 12), (0, 10), (20, 25)]), 17)
        self.assertEqual(union_duration([]), 0)

    def test_trace_does_not_count_cpu_scopes_as_gpu(self):
        def event(cat, name, ts, dur):
            return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}
        result = summarize_trace({"traceEvents": [
            event("kernel", "A", 0, 10),
            event("kernel", "B", 5, 10),
            event("gpu_memcpy", "copy", 20, 5),
            event("cpu_op", "aten::matmul", 0, 1000),
            event("cuda_runtime", "cudaLaunchKernel", 0, 4),
        ]}, steps=2)
        self.assertEqual(result["kernels_per_forward"], 1)
        self.assertAlmostEqual(result["summed_kernel_ms_per_forward"], .01)
        self.assertAlmostEqual(result["observed_gpu_active_union_ms"], .02)
        self.assertAlmostEqual(result["observed_gpu_gap_ms"], .005)
        self.assertEqual(result["cuda_runtime_calls"]["cudaLaunchKernel"], 1)

    def test_missing_cuda_trace_is_not_zero_cost(self):
        result = summarize_trace({"traceEvents": []}, steps=2)
        self.assertFalse(result["gpu_trace_available"])
        self.assertIsNone(result["summed_kernel_ms_per_forward"])
        self.assertIsNone(result["observed_gpu_gap_ms"])


if __name__ == "__main__":
    unittest.main()
