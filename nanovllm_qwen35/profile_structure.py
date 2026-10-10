"""Separate diagnostic run; does not edit production engine or report benchmark speed."""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import textwrap
from unittest.mock import patch


def annotate_runner(module, record_function):
    """Instrument the existing run body in memory, preserving its exception handling."""
    original = inspect.unwrap(module.GPUExecutionRunner.run)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    execution_try = next(node for node in function.body if isinstance(node, ast.Try))
    body = execution_try.body
    # Refuse unknown source instead of silently profiling incorrect stage boundaries.
    begin = next(i for i, node in enumerate(body)
                 if isinstance(node, ast.If) and 'logits.ndim' in ast.unparse(node.test))
    assert ast.unparse(body[-1]).endswith('states.synchronize()')
    def region(label, statements):
        return ast.With(items=[ast.withitem(context_expr=ast.Call(
            func=ast.Name(id='record_function', ctx=ast.Load()),
            args=[ast.Constant(label)], keywords=[]))], body=statements)
    sampling = []
    for node in body[begin:]:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tokens' for t in node.targets):
            assert 'ids.tolist()' in ast.unparse(node)
            node = region('sampling/token_ids_to_host', [node])
        sampling.append(node)
    execution_try.body = body[:begin] + [region('sampling_sync', sampling)]
    ast.fix_missing_locations(tree)
    namespace = dict(vars(module), record_function=record_function)
    exec(compile(tree, '<profiling-only-runner>', 'exec'), namespace)
    return namespace['run']


def summarize_trace(path):
    events = json.loads(path.read_text())['traceEvents']
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    steps = [e for e in complete if e.get('cat') == 'user_annotation' and e['name'].startswith('step/')]
    summaries = []
    for step in steps:
        start, end = step['ts'], step['ts'] + step['dur']
        inside = [e for e in complete if start <= e['ts'] < end]
        annotations = defaultdict(list)
        for e in inside:
            if e.get('cat') == 'user_annotation':
                annotations[e['name']].append(e['dur'])
        device = [e for e in inside if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
        merged = []
        for e in sorted(device, key=lambda x: x['ts']):
            a, b = max(start, e['ts']), min(end, e['ts'] + e['dur'])
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        gaps = sorted([(b[0]-a[1], a[1], b[0]) for a,b in zip(merged,merged[1:])], reverse=True)
        summaries.append({
            'step': step['name'], 'instrumented_host_ms': step['dur']/1000,
            'annotations_cpu_inclusive_ms': {k: {'count': len(v), 'sum_ms': sum(v)/1000} for k,v in annotations.items()},
            'device_event_count': len(device),
            'device_activity_union_ms': sum(b-a for a,b in merged)/1000,
            'internal_device_gaps_top5_us': gaps[:5],
            'memcpy_events': dict(Counter(e['name'] for e in device if e.get('cat') == 'gpu_memcpy')),
            'synchronization_api_counts': dict(Counter(e['name'] for e in inside if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e['name'])),
        })
    return {'steps': summaries, 'cuda_kernel_events': sum(e.get('cat') == 'kernel' for e in complete),
            'caution': 'Instrumented timings, not baseline. Nested CPU ranges overlap; GPU gaps alone do not establish cause.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/ubuntu/huggingface/Qwen3.5-4B')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from nanovllm import SamplingParams
    import qwen35_adapter.gpu_engine as gpu
    import qwen35_adapter.pool_backend as pool
    from qwen35_adapter.runner_inputs import RunnerInputBuilder
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    engine = gpu.Qwen35Engine(args.model, max_model_len=512, max_num_seqs=2, block_size=256)
    base = engine.tokenizer('Explain how KV cache accelerates inference.', add_special_tokens=False)['input_ids']
    prompts = [(base*20)[i:i+64] for i in range(2)]
    sp = SamplingParams(max_tokens=3, ignore_eos=True)
    report = {'status': 'running', 'cases': [], 'torch': torch.__version__,
              'scope': 'B=1/2,L=64,one prefill and two decode steps each; two warmup requests per case',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(gpu.__file__), Path(pool.__file__))}}
    layer_rows = Counter()
    def wrap(label, fn):
        def call(*a, **kw):
            with record_function(label):
                return fn(*a, **kw)
        return call
    original_layer = pool.decoder_layer
    def layer(*a, **kw):
        layer_id = a[4]
        row = layer_rows[layer_id]
        layer_rows[layer_id] += 1
        with record_function(f'layer/{layer_id}/request_row/{row}'):
            return original_layer(*a, **kw)
    original_forward = engine.backend.forward
    def forward(*a, **kw):
        layer_rows.clear()
        with record_function('forward'):
            return original_forward(*a, **kw)
    try:
        for batch_size in (1, 2):
            for _ in range(2):
                expected = engine.generate(prompts[:batch_size], sp)
            torch.cuda.synchronize()
            requests = [engine.add_request(p, sp) for p in prompts[:batch_size]]
            with ExitStack() as stack:
                stack.enter_context(patch.object(gpu.GPUExecutionRunner, 'run', annotate_runner(gpu, record_function)))
                for obj, name, label in (
                    (engine.scheduler, 'schedule', 'schedule'),
                    (RunnerInputBuilder, 'prepare', 'prepare'),
                    (engine.backend, 'compute_logits', 'LM_Head'),
                    (engine.scheduler, 'postprocess', 'postprocess'),
                    (engine.states, 'synchronize', 'explicit_device_sync'),
                ):
                    stack.enter_context(patch.object(obj, name, wrap(label, getattr(obj, name))))
                stack.enter_context(patch.object(engine.backend, 'forward', forward))
                stack.enter_context(patch.object(pool, 'decoder_layer', layer))
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=False, profile_memory=False, with_stack=False) as prof:
                    for step in range(3):
                        with record_function(f'step/B{batch_size}/{"prefill" if step==0 else "decode"}/{step}'):
                            engine.step()
                        prof.step()
                    torch.cuda.synchronize()
            assert engine.scheduler.is_finished()
            actual = [list(s.completion_token_ids) for s in requests]
            assert actual == [r['token_ids'] for r in expected], 'Instrumentation changed output'
            path = output / f'batch{batch_size}.trace.json'
            prof.export_chrome_trace(str(path))
            summary = summarize_trace(path)
            if not summary['cuda_kernel_events']:
                raise RuntimeError('No CUDA kernels captured; do not infer GPU gaps from CPU-only trace')
            report['cases'].append({'batch': batch_size, 'tokens_equal_unprofiled': True, **summary})
            print(f'B={batch_size}: captured {summary["cuda_kernel_events"]} kernels', flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not engine.runner.poisoned and not engine.states.poisoned and engine.scheduler._pending is None:
            engine.close()


if __name__ == '__main__':
    main()
