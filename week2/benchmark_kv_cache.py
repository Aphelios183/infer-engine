"""Lesson 3：完整前缀重算 vs KV Cache。默认只展示计划；加 --run 才运行。

本实验是单层教学 MHA，不包含 tokenizer、采样、网络或请求调度。
使用参考 KVCache，不会自动切换到 kv_cache_exercise.py。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

try:
    from .kv_cache import KVCache, MultiHeadAttention, forward_cached
except ImportError:
    from kv_cache import KVCache, MultiHeadAttention, forward_cached


class Case:
    def __init__(self, prompt, steps, batch, dim, heads, device, seed):
        if any(type(x) is not int or x <= 0 for x in (prompt, steps, batch, dim, heads)):
            raise ValueError("长度、步数、batch、dim 和 heads 必须是正整数")
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        self.prompt, self.steps = prompt, steps
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = MultiHeadAttention(dim, heads).to(self.device).eval()
        self.x = torch.randn(batch, prompt + steps, dim, device=self.device)
        self.cache = KVCache(batch, heads, prompt + steps, dim // heads,
                             device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def check_correctness(self):
        """逐步对拍，误差归约/CPU 同步只出现在计时外。"""
        self.cache.reset()
        expected = self.model(self.x[:, :self.prompt])
        actual = forward_cached(self.model, self.x[:, :self.prompt], self.cache)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        error = (actual - expected).abs().max().item()
        for end in range(self.prompt + 1, self.prompt + self.steps + 1):
            expected = self.model(self.x[:, :end])[:, -1:]
            actual = forward_cached(self.model, self.x[:, end-1:end], self.cache)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            error = max(error, (actual - expected).abs().max().item())
            if self.cache.length != end:
                raise AssertionError("缓存长度没有随 Decode 正确增长")
        return {"max_abs_error": error, "decode_checks": self.steps,
                "atol": 1e-5, "rtol": 1e-4}

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timed(self, function):
        self.synchronize()
        start = time.perf_counter()
        function()
        self.synchronize()
        return (time.perf_counter() - start) * 1000

    @torch.inference_mode()
    def run_round(self, path):
        if path not in ("baseline", "cache"):
            raise ValueError("未知路径")
        if path == "cache":
            # 每轮从空缓存开始，reset 不计时；Prefill 写 KV 计入 Prefill。
            self.cache.reset()

        def prefill():
            if path == "baseline":
                return self.model(self.x[:, :self.prompt])
            return forward_cached(self.model, self.x[:, :self.prompt], self.cache)

        def decode():
            # 整个 N 步为一个测量区间，不在每个 token 后同步。
            for end in range(self.prompt + 1, self.prompt + self.steps + 1):
                if path == "baseline":
                    output = self.model(self.x[:, :end])[:, -1:]
                else:
                    # 包含新 Q/K/V 投影、append、历史读取和输出投影。
                    output = forward_cached(self.model, self.x[:, end-1:end], self.cache)
            return output

        prefill_ms = self.timed(prefill)
        decode_ms = self.timed(decode)
        if path == "cache" and self.cache.length != self.prompt + self.steps:
            raise AssertionError("每轮缓存最终长度错误")
        return {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
                "decode_step_mean_ms": decode_ms / self.steps,
                "total_ms": prefill_ms + decode_ms}


def summarize(rows):
    """先在每轮求 Prefill+Decode，再取中位数，不相加两个中位数。"""
    return {key: {"median": statistics.median(row[key] for row in rows),
                  "min": min(row[key] for row in rows),
                  "max": max(row[key] for row in rows)}
            for key in ("prefill_ms", "decode_ms", "decode_step_mean_ms", "total_ms")}


def measure_case(case, warmup, repeats):
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup 必须非负，repeats 必须为正")
    correctness = case.check_correctness()
    for _ in range(warmup):
        case.run_round("baseline")
        case.run_round("cache")
    raw = {"baseline": [], "cache": []}
    orders = []
    for index in range(repeats):
        order = ["baseline", "cache"] if index % 2 == 0 else ["cache", "baseline"]
        orders.append(order)
        for path in order:
            raw[path].append(case.run_round(path))
    summary = {path: summarize(rows) for path, rows in raw.items()}
    speedup = {key: summary["baseline"][key]["median"] / summary["cache"][key]["median"]
               for key in ("prefill_ms", "decode_ms", "total_ms")}
    return {"prompt": case.prompt, "steps": case.steps, "correctness": correctness,
            "raw": raw, "orders": orders, "summary": summary,
            "speedup_ratio_of_medians": speedup,
            "kv_used_bytes_at_end": case.cache.used_bytes,
            "kv_allocated_bytes": case.cache.allocated_bytes}


def gpu_snapshot():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
             "--format=csv"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="显式开始实验；否则仅输出计划")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512, 1024])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="JSON 原始数据文件；拒绝覆盖已有文件")
    args = parser.parse_args()
    if (min(args.lengths + [args.steps, args.batch, args.dim, args.heads,
                           args.repeats, args.cpu_threads]) <= 0 or args.warmup < 0):
        parser.error("维度/步数/重复次数/线程数必须为正，warmup 必须非负")
    if args.dim % args.heads:
        parser.error("dim 必须能被 heads 整除")
    config = {**vars(args), "output": str(args.output) if args.output else None,
              "dtype": "float32", "cache_implementation": "reference KVCache",
              "timer": "synchronized wall-clock per phase; no per-step sync"}
    if not args.run:
        print(json.dumps({"status": "planned_only", "config": config,
                          "note": "未执行模型或计时；确认 GPU 空闲后加 --run。"},
                         ensure_ascii=False, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(__file__).parent / "results" / f"kv_cache_{stamp}.json"
    if output.exists():
        parser.error(f"拒绝覆盖已有结果：{output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    torch.set_num_threads(args.cpu_threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__).resolve(), root / "week2/kv_cache.py",
                               root / "week1/mha.py")}
    metadata = {"started_utc": stamp, "python": platform.python_version(),
                "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cpu_threads": torch.get_num_threads(), "argv": sys.argv,
                "source_sha256": source_hashes,
                "gpu_before": gpu_snapshot() if args.device == "cuda" else None}
    results = []
    print("L | full prefill | cache prefill | full decode | cache decode | decode speedup")
    for length in args.lengths:
        case = Case(length, args.steps, args.batch, args.dim, args.heads, args.device, args.seed)
        result = measure_case(case, args.warmup, args.repeats)
        results.append(result)
        a, b = result["summary"]["baseline"], result["summary"]["cache"]
        print(f"{length} | {a['prefill_ms']['median']:.4f} | {b['prefill_ms']['median']:.4f} | "
              f"{a['decode_ms']['median']:.4f} | {b['decode_ms']['median']:.4f} | "
              f"{result['speedup_ratio_of_medians']['decode_ms']:.3f}x", flush=True)
        del case
    metadata["gpu_after"] = gpu_snapshot() if args.device == "cuda" else None
    payload = {"config": config, "metadata": metadata, "results": results,
               "limitations": ["single-layer dense MHA, synthetic hidden states, not a serving benchmark",
                               "allocation/model/input setup and reset excluded",
                               "total_ms is sum of separately synchronized phase times, not unsplit E2E",
                               "GPU snapshots do not prove exclusive access during the experiment",
                               "CPU timing is not L40 performance; no profiler during timing"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"原始数据与环境：{output.resolve()}")


if __name__ == "__main__":
    main()
