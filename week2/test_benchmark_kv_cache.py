"""Lesson 3 benchmark 的工作负载、重置和统计口径测试（CPU）。"""
import importlib.util
import unittest

import torch


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        # 缺失实现时给出明确失败，而不是导入阶段崩溃。
        self.assertIsNotNone(importlib.util.find_spec("week2.benchmark_kv_cache"),
                             "尚未实现 benchmark_kv_cache")
        from week2 import benchmark_kv_cache as bench
        self.bench = bench

    def make_case(self):
        return self.bench.Case(prompt=3, steps=2, batch=2, dim=8, heads=2,
                               device="cpu", seed=7)

    def test_prefill_and_all_decode_outputs_match(self):
        case = self.make_case()
        errors = case.check_correctness()
        self.assertEqual(errors["decode_checks"], 2)
        self.assertLess(errors["max_abs_error"], 1e-5)

    def test_workload_uses_full_prefix_or_one_new_token(self):
        # 把 baseline 改成只输入新 token，或 cache 重复输入历史，应失败。
        case = self.make_case()
        seen = []
        handle = case.model.q_proj.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].size(1)))
        try:
            case.run_round("baseline")
            self.assertEqual(seen, [3, 4, 5])
            seen.clear()
            case.run_round("cache")
            self.assertEqual(seen, [3, 1, 1])
        finally:
            handle.remove()

    def test_each_cache_round_resets_and_counts_decode_writes(self):
        case = self.make_case()
        for _ in range(4):
            row = case.run_round("cache")
            self.assertEqual(case.cache.length, 5)
            self.assertGreater(row["prefill_ms"], 0)
            self.assertGreater(row["decode_ms"], 0)
            self.assertAlmostEqual(row["total_ms"], row["prefill_ms"] + row["decode_ms"])
            self.assertAlmostEqual(row["decode_step_mean_ms"], row["decode_ms"] / 2)

    def test_measured_rows_exclude_warmup(self):
        result = self.bench.measure_case(self.make_case(), warmup=2, repeats=3)
        self.assertEqual(len(result["raw"]["baseline"]), 3)
        self.assertEqual(len(result["raw"]["cache"]), 3)
        self.assertEqual(result["orders"], [["baseline", "cache"],
                                           ["cache", "baseline"], ["baseline", "cache"]])

    def test_total_summary_uses_per_round_sums(self):
        # median(prefill)+median(decode)=12，但 median(total)=11。
        rows = [{"prefill_ms": a, "decode_ms": b, "total_ms": a+b,
                 "decode_step_mean_ms": b/2} for a,b in [(1,10),(2,100),(3,1)]]
        summary = self.bench.summarize(rows)
        self.assertEqual(summary["total_ms"]["median"], 11)
        self.assertEqual(summary["decode_ms"]["min"], 1)
        self.assertEqual(summary["decode_ms"]["max"], 100)

    def test_invalid_workloads_are_rejected(self):
        for kwargs in [{"prompt": 0}, {"steps": 0}, {"dim": 7}, {"batch": -1}]:
            params = dict(prompt=3, steps=2, batch=2, dim=8, heads=2, device="cpu", seed=7)
            params.update(kwargs)
            with self.assertRaises(ValueError):
                self.bench.Case(**params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
