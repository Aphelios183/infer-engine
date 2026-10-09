"""Unprofiled, fixed-token, offline closed-batch GPU baseline (not an HTTP SLA)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)


def source_digest():
    paths = list(Path('qwen35_adapter').glob('*.py')) + list(Path('nanovllm').rglob('*.py'))
    paths.append(Path(__file__))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run_case(engine, prompts, new_tokens, torch, SamplingParams):
    assert engine.scheduler.is_finished() and not engine.runner.capture_logits
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    start = time.perf_counter()
    requests = [engine.add_request(ids, SamplingParams(max_tokens=new_tokens, ignore_eos=True))
                for ids in prompts]
    emissions = {s.seq_id: [] for s in requests}
    step_ms = []
    while not engine.scheduler.is_finished():
        before = time.perf_counter()
        result = engine.step()
        # The public runner already synchronizes before committing tokens.
        now = time.perf_counter()
        if result is None:
            raise RuntimeError('No progress while requests remain')
        step_ms.append({'is_prefill': result.is_prefill, 'ms': (now - before) * 1000})
        for request_id in result.request_ids:
            emissions[request_id].append(now)
    end = time.perf_counter()
    torch.cuda.synchronize()
    per_request = []
    for seq in requests:
        times = emissions[seq.seq_id]
        assert len(times) == len(seq.completion_token_ids) == new_tokens
        assert engine.scheduler.completed[seq.seq_id].reason == 'length'
        per_request.append({
            'ttft_ms': (times[0] - start) * 1000,
            'tpot_ms': (times[-1] - times[0]) * 1000 / (new_tokens - 1),
            'itl_ms': [(b - a) * 1000 for a, b in zip(times, times[1:])],
            'output_ids': list(seq.completion_token_ids),
        })
    total = len(prompts) * new_tokens
    return {
        'requests': per_request, 'steps': step_ms, 'elapsed_s': end - start,
        'output_tokens_per_s': total / (end - start),
        'decode_tokens_per_s': len(prompts) * (new_tokens - 1) /
            (max(t[-1] for t in emissions.values()) - min(t[0] for t in emissions.values())),
        'input_plus_output_tokens_per_s': (sum(map(len, prompts)) + total) / (end - start),
        'resident_allocated_bytes': resident,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/ubuntu/huggingface/Qwen3.5-4B')
    parser.add_argument('--lengths', nargs='+', type=int, default=[19, 64, 128, 256])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', default='performance_baseline.json')
    parser.add_argument('--monitor-gpu', default=None, help='Physical GPU index or UUID for nvidia-smi 1s sampling')
    args = parser.parse_args()
    if (any(x < 1 or x + args.new_tokens > 512 for x in args.lengths)
            or any(x not in (1, 2) for x in args.batches)
            or args.new_tokens < 2 or args.warmup < 1 or args.repeats < 3):
        parser.error('Require lengths+new_tokens<=512, batch 1/2, new_tokens>=2, warmup>=1, repeats>=3')
    output = Path(args.output)
    report = output.with_suffix('.md')
    monitor_path = output.with_suffix('.gpu.csv')
    if output.exists() or report.exists() or monitor_path.exists():
        parser.error('Choose a fresh output path; existing measurements are not overwritten')
    import torch
    import transformers
    from nanovllm.qwen35 import Qwen35LLM
    from nanovllm import SamplingParams
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    gpu_query = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv']
    data = {'status': 'running', 'args': vars(args), 'environment': {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(0), 'gpu_before': command(gpu_query),
        'driver': command(['nvidia-smi', '--query-gpu=index,driver_version', '--format=csv']),
        'processes_before': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv']),
        'pid': os.getpid(),
        'git_head': command(['git', 'rev-parse', 'HEAD']),
        'git_status': command(['git', 'status', '--short']), 'source_sha256': source_digest(),
        'tf32': False, 'cpu_threads': 2, 'capture_logits': False,
        'max_model_len': 512, 'max_num_seqs': 2, 'block_size': 256,
    }, 'cases': []}
    engine = None
    monitor = None
    monitor_file = None
    try:
        if args.monitor_gpu is not None:
            monitor_file = monitor_path.open('x', encoding='utf-8')
            monitor = subprocess.Popen(['nvidia-smi', '-i', args.monitor_gpu,
                '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm',
                '--format=csv', '--loop=1'], stdout=monitor_file, stderr=subprocess.STDOUT)
        engine = Qwen35LLM(args.model, max_model_len=512, max_num_seqs=2, block_size=256, greedy=True)
        seed = engine.tokenizer('Explain how a key value cache accelerates language model inference.',
                                add_special_tokens=False)['input_ids']
        for length in args.lengths:
            for batch in args.batches:
                # Exact-length synthetic token workload, not a quality evaluation.
                prompts = [(seed * ((length + i) // len(seed) + 2))[i:i + length]
                           for i in range(batch)]
                for _ in range(args.warmup):
                    run_case(engine, prompts, args.new_tokens, torch, SamplingParams)
                runs = [run_case(engine, prompts, args.new_tokens, torch, SamplingParams)
                        for _ in range(args.repeats)]
                case = {'prompt_length': length, 'batch': batch, 'input_ids': prompts, 'runs': runs}
                data['cases'].append(case)
                output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'L={length} B={batch}: median output '
                      f'{statistics.median(r["output_tokens_per_s"] for r in runs):.2f} tok/s', flush=True)
        data['status'] = 'complete'
    except BaseException as exc:
        data['status'] = 'failed'
        data['error'] = repr(exc)
        raise
    finally:
        data['environment']['gpu_after_measurement'] = command(gpu_query)
        data['environment']['processes_after_measurement'] = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'])
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if monitor_file is not None:
            monitor_file.close()
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        if engine is not None and not engine.runner.poisoned and not engine.states.poisoned:
            engine.close()
    lines = ['# GPU performance baseline', '',
        'Offline fixed-length closed batches; greedy, ignore_eos=True. No profiler, HF comparison or snapshots.',
        'TTFT starts before enqueue and ends at first settled token. Tokenization, model load and text decoding excluded.',
        'TPOT=(last-first token time)/(N-1). Output throughput=B*N/end-to-end batch seconds.',
        'Values are medians across repetitions; TTFT/TPOT first averaged over requests per repetition.',
        'P95 is descriptive only with few repeats. Peaks include model, pools and temporary tensors.',
        'Reserved is allocator memory, NOT logical cache size. Warm allocator retained between cases.', '',
        '| L | B | TTFT median/p95 ms | TPOT median/p95 ms | Output tok/s median [min,max] | Decode tok/s | Peak allocated GiB | Peak reserved GiB |',
        '|---|---|---|---|---|---|---|---|']
    for c in data['cases']:
        runs = c['runs']
        ttft = [statistics.mean(s['ttft_ms'] for s in r['requests']) for r in runs]
        tpot = [statistics.mean(s['tpot_ms'] for s in r['requests']) for r in runs]
        rate = [r['output_tokens_per_s'] for r in runs]
        lines.append(f'| {c["prompt_length"]} | {c["batch"]} | '
                     f'{statistics.median(ttft):.2f}/{percentile(ttft,.95):.2f} | '
                     f'{statistics.median(tpot):.2f}/{percentile(tpot,.95):.2f} | '
                     f'{statistics.median(rate):.2f} [{min(rate):.2f},{max(rate):.2f}] | '
                     f'{statistics.median(r["decode_tokens_per_s"] for r in runs):.2f} | '
                     f'{max(r["peak_allocated_bytes"] for r in runs)/2**30:.3f} | '
                     f'{max(r["peak_reserved_bytes"] for r in runs)/2**30:.3f} |')
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
