"""Week 1: measure first, then explain MHA overhead. No extra dependencies.

Run with the existing nanovllm environment on an idle GPU. See PROFILING.md.
All latency numbers come from runs WITHOUT the profiler. Trace durations are
diagnostic observations, never substituted for those baseline measurements.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

try:
    from mha import MultiHeadAttention, build_causal_mask
except ImportError:
    from week1.mha import MultiHeadAttention, build_causal_mask


def attention_forward(model, x, *, cached_mask=None, sdpa=False, annotate=False):
    """Same weights/layout as mha.py; optional controlled changes and labels."""
    def scope(name):
        return record_function(name) if annotate else nullcontext()

    with scope("01_QKV_projection_and_split"):
        q = model._split_heads(model.q_proj(x))
        k = model._split_heads(model.k_proj(x))
        v = model._split_heads(model.v_proj(x))
    if sdpa:
        with scope("02_SDPA_auto_backend"):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    else:
        with scope("02_QK_and_scale"):
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        with scope("03_mask_build_or_reuse"):
            mask = cached_mask
            if mask is None:
                mask = build_causal_mask(x.size(1), device=x.device)
        with scope("04_masked_fill"):
            scores = scores.masked_fill(mask, float("-inf"))
        with scope("05_softmax"):
            weights = F.softmax(scores, dim=-1)
        with scope("06_attention_times_V"):
            out = weights @ v
    with scope("07_merge_heads_contiguous"):
        out = model._merge_heads(out)
    with scope("08_output_projection"):
        return model.out_proj(out)


def measure(fn, *, warmup, iters, repeats):
    """Pipeline average, not isolated single-request latency.

    CUDA events bracket a stream interval; that interval can include idle gaps
    while the CPU prepares the next launch. It is NOT the sum of kernel times.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()  # initialize events outside the measured loop
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        submit_start = time.perf_counter()
        for _ in range(iters):
            fn()
        submit_end = time.perf_counter()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter()
        samples.append({
            "wall_ms": (wall_end - wall_start) * 1000 / iters,
            "host_submit_ms": (submit_end - submit_start) * 1000 / iters,
            "stream_interval_ms": start.elapsed_time(end) / iters,
            "tail_wait_ms": (wall_end - submit_end) * 1000 / iters,
        })
    return {
        "median": {key: statistics.median(s[key] for s in samples) for key in samples[0]},
        "wall_min_ms": min(s["wall_ms"] for s in samples),
        "wall_max_ms": max(s["wall_ms"] for s in samples),
        "samples": samples,
    }


def measure_sync_each(fn, *, iters):
    """A separate control: synchronize after EVERY forward (includes sync cost)."""
    samples = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iters)
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "iters": iters}


def capture_graph(fn):
    # Capture the original MHA, including mask construction. Warm up on a side
    # stream; model/x/output remain alive until replay measurements finish.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay():
        graph.replay()
        return output

    return replay, graph


def union_duration(intervals):
    """Union, rather than sum, prevents double counting overlapping streams."""
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if end is None or left >= end:
            total += right - left
        elif right > end:
            total += right - end
        end = right if end is None else max(end, right)
    return total


def summarize_trace(document, steps):
    events = [e for e in document.get("traceEvents", [])
              if e.get("ph") == "X" and isinstance(e.get("dur"), (float, int))]
    kernels = [e for e in events if e.get("cat", "").lower() == "kernel"]
    gpu_events = [e for e in events
                  if e.get("cat", "").lower() in {"kernel", "gpu_memcpy", "gpu_memset"}]
    by_name = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for e in kernels:
        by_name[e["name"]]["count"] += 1
        by_name[e["name"]]["total_us"] += e["dur"]
    runtime = Counter(e.get("name", "") for e in events
                      if e.get("cat", "").lower() == "cuda_runtime")
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_events if e.get("dur", 0) > 0]
    span = max(b for _, b in intervals) - min(a for a, _ in intervals) if intervals else None
    active = union_duration(intervals) if intervals else None
    return {
        "gpu_trace_available": bool(kernels),
        "profiled_forwards": steps,
        "kernel_count": len(kernels),
        "kernels_per_forward": len(kernels) / steps if kernels else None,
        "summed_kernel_ms_per_forward": sum(e["dur"] for e in kernels) / 1000 / steps if kernels else None,
        "observed_gpu_span_ms": span / 1000 if span is not None else None,
        "observed_gpu_active_union_ms": active / 1000 if active is not None else None,
        "observed_gpu_gap_ms": max(0.0, span - active) / 1000 if span is not None else None,
        "cuda_runtime_calls": dict(runtime),
        "top_kernels": sorted(
            [{"name": name, **v} for name, v in by_name.items()],
            key=lambda item: item["total_us"], reverse=True)[:12],
        "interpretation": "Profiler-instrumented trace only. Gaps do not by themselves prove CPU launch bottlenecks; timing perturbation and other GPU users are possible.",
    }


def collect_profile(fn, trace_fn, *, steps, output, name):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=False, with_stack=False) as prof:
        for index in range(steps):
            with record_function(f"forward_{index}"):
                trace_fn()
        torch.cuda.synchronize()
    trace_path = output / f"{name}.trace.json"
    prof.export_chrome_trace(str(trace_path))
    averages = prof.key_averages(group_by_input_shape=True)
    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=20)
    summary = summarize_trace(json.loads(trace_path.read_text(encoding="utf-8")), steps)
    # Kineto can expose device-side user annotations in key_averages; those are
    # enclosing ranges, not kernels, and can misleadingly exceed 100% in a
    # GPU-sorted table. Aggregate raw cat=kernel events for the GPU table.
    gpu_lines = ["Count    Total(us)    Mean(us)    Kernel"]
    for item in summary["top_kernels"]:
        gpu_lines.append(f"{item['count']:5d} {item['total_us']:12.3f} {item['total_us'] / item['count']:11.3f}    {item['name']}")
    gpu_table = "\n".join(gpu_lines)
    (output / f"{name}.operators.txt").write_text(
        "CPU SELF TIME (not end-to-end latency)\n" + cpu_table +
        "\nGPU KERNEL EVENTS ONLY (user ranges excluded; not wall-clock latency)\n" + gpu_table, encoding="utf-8")
    summary["trace_file"] = trace_path.name
    summary["operators_file"] = f"{name}.operators.txt"
    return summary


def command_output(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def gpu_snapshot():
    return command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                           "--format=csv,noheader"])


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def write_results(result, output):
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Week 1 MHA 性能实验", "", "耗时来自关闭 profiler 的实验。单位 ms/forward；每项报告重复测量的中位数。", "",
             "| T | 版本 | Wall | CPU 提交 | GPU stream 区间 | 尾部等待 | 相对 eager 加速 |",
             "|---|---|---|---|---|---|---|"]
    for case in result["cases"]:
        baseline = case["variants"]["eager"]["timing"]["median"]["wall_ms"]
        for name, data in case["variants"].items():
            if "timing" not in data:
                lines.extend([f"<!-- T={case['seq_len']} {name}: unavailable; see summary.json -->"])
                continue
            m = data["timing"]["median"]
            lines.append(f"| {case['seq_len']} | {name} | {fmt(m['wall_ms'])} | {fmt(m['host_submit_ms'])} | {fmt(m['stream_interval_ms'])} | {fmt(m['tail_wait_ms'])} | {baseline / m['wall_ms']:.2f}x |")
    lines += ["", "## 如何解释", "",
              "- CPU 提交与 GPU 执行重叠，不能相加，也不能用两者差值推算 launch 开销。",
              "- GPU Event 的 stream 区间可以包含等 CPU 发起下一次操作的空档，并非纯 kernel 时间。",
              "- cached_mask 仅移出 mask 构造；仍然计算 scores、masked_fill、softmax 和 AV。",
              "- CUDA Graph 保留原计算图，但改变提交和分配方式；其收益不能全部归为某一次 cudaLaunchKernel 调用。",
              "- SDPA 后端由 PyTorch 自动选择，查看 trace 中的算子/kernel 名称，不预设使用 FlashAttention。",
              "- profiler trace 只用于解释调用和执行分布，不替代上表。", "",
              "## profiler 的诊断观察（受采集扰动）", "",
              "| T | 版本 | kernels/forward | kernel 时长之和/forward (ms) | trace 内 GPU 空档 (ms) |",
              "|---|---|---|---|---|"]
    for case in result["cases"]:
        for name, data in case["variants"].items():
            p = data.get("profile")
            if p:
                lines.append(f"| {case['seq_len']} | {name} | {fmt(p['kernels_per_forward'])} | {fmt(p['summed_kernel_ms_per_forward'])} | {fmt(p['observed_gpu_gap_ms'])} |")
    lines += ["", "GPU 空档按 trace 首个到最后一个 GPU 活动之间的区间计算；不是设备全局利用率。",
              "完整 samples、逐次同步对照、正确性误差、环境、GPU 前后快照见 summary.json。",
              "分析顺序和练习见 week1/PROFILING.md。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    parser.add_argument("--profile-seq-lens", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None, help="Default preserves the current PyTorch CPU thread count")
    parser.add_argument("--skip-profiler", action="store_true", help="Run only uninstrumented latency comparisons")
    parser.add_argument("--output", type=Path, default=None, help="A new, non-existing directory; defaults below results/week1")
    args = parser.parse_args()
    for name in ["batch", "hidden", "heads", "warmup", "iters", "repeats", "profile_steps"]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.hidden % args.heads or any(t <= 0 for t in args.seq_lens + args.profile_seq_lens):
        parser.error("hidden must be divisible by heads, and sequence lengths must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("threads must be positive")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU。先选择空闲 GPU，再执行 CUDA_VISIBLE_DEVICES=2 python week1/profile_attention.py")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or root / "results" / "week1" / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    result = {
        "schema_version": 1, "status": "running", "run_id": run_id, "config": config,
        "environment": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": "float32", "num_threads": torch.get_num_threads(), "tf32_matmul": False,
            "git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=root),
            "git_status": command_output(["git", "status", "--short"], cwd=root),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path(__file__).with_name("mha.py")]},
            "gpu_before": gpu_snapshot()},
        "cases": [],
    }
    print(f"GPU: {result['environment']['device']}; CPU threads: {torch.get_num_threads()}; FP32", flush=True)
    print(f"Output: {output}", flush=True)
    for seq_len in args.seq_lens:
        model = MultiHeadAttention(args.hidden, args.heads).cuda().eval()
        x = torch.randn(args.batch, seq_len, args.hidden, device="cuda")
        mask = build_causal_mask(seq_len, device=x.device)
        original = lambda: model(x)
        expected = original().clone()
        variant_fns = {
            "eager": original,
            "cached_mask": lambda: attention_forward(model, x, cached_mask=mask),
            "sdpa": lambda: attention_forward(model, x, sdpa=True),
        }
        traced_fns = {
            "eager": lambda: attention_forward(model, x, annotate=True),
            "cached_mask": lambda: attention_forward(model, x, cached_mask=mask, annotate=True),
            "sdpa": lambda: attention_forward(model, x, sdpa=True, annotate=True),
        }
        case = {"seq_len": seq_len, "variants": {}}
        result["cases"].append(case)
        graph = None
        try:
            replay, graph = capture_graph(original)
            variant_fns["cuda_graph"] = replay
            traced_fns["cuda_graph"] = replay
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            case["variants"]["cuda_graph"] = {"status": "unavailable", "reason": str(error)}
        # Validate instrumented decomposition too, before timing anything.
        torch.testing.assert_close(attention_forward(model, x), expected, atol=1e-5, rtol=1e-4)
        for name, fn in variant_fns.items():
            actual = fn()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            data = {"status": "ok", "max_abs_error": (actual - expected).abs().max().item()}
            data["timing"] = measure(fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
            data["sync_each"] = measure_sync_each(fn, iters=min(args.iters, 20))
            case["variants"][name] = data
            m = data["timing"]["median"]
            print(f"T={seq_len:4d} {name:12s} wall={m['wall_ms']:.4f} ms submit={m['host_submit_ms']:.4f} stream={m['stream_interval_ms']:.4f}", flush=True)
        # All benchmark variants finish before any profiler collection at this T.
        if not args.skip_profiler and seq_len in args.profile_seq_lens:
            for name, fn in variant_fns.items():
                case["variants"][name]["profile"] = collect_profile(
                    fn, traced_fns[name], steps=args.profile_steps,
                    output=output, name=f"T{seq_len}_{name}")
        write_results(result, output)
        torch.cuda.synchronize()
        variant_fns.clear()
        traced_fns.clear()
        # Capture output and graph live until this sequence length is complete.
        del graph, model, x, mask, expected, actual, fn, original
        if "replay" in locals():
            del replay
        torch.cuda.empty_cache()
    result["environment"]["gpu_after"] = gpu_snapshot()
    result["status"] = "complete"
    write_results(result, output)
    print("完成：先读 report.md，再对照 operators.txt 和 trace.json；详细计时样本见 summary.json。", flush=True)


if __name__ == "__main__":
    main()
