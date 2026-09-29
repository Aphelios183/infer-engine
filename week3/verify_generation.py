"""Fixed-prefix Qwen3.5 verification. No free-generation text equivalence claims."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import torch
from week3.reference_runtime import forward_last, load_reference, validate_ids
from week3.cache_inspection import snapshot_cache
from week3.inspect_model import inspect_prompt
from week3.minimal_generate import run_greedy

METRICS = ('max_abs', 'mean_abs', 'relative_l2')
CALIBRATION = ['简要解释矩阵乘法。', 'Name two primary colors.']
ACCEPTANCE = ['用一句话解释 KV Cache。', '计算 2+3，只输出结果。', 'What is prefill in language model inference?']
CONTINUATION = '缓存复用历史信息。'


def compare_logits(reference, candidate):#数值比较函数，比较输出得分的一致性
    for x in (reference, candidate):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != 1
                or x.shape[1] < 2 or not torch.isfinite(x).all().item()):
            raise ValueError('expected finite [1,V] logits, V>=2')
    if reference.shape != candidate.shape:
        raise ValueError('logits shape mismatch')
    r, c = reference.float(), candidate.float()
    diff = c - r
    norm = torch.linalg.vector_norm(r).item()
    error = torch.linalg.vector_norm(diff).item()
    if not all(math.isfinite(x) for x in (norm, error)):
        raise ValueError('nonfinite error metric')
    def margin(x):
        t = x.topk(2, dim=-1).values
        return (t[0, 0] - t[0, 1]).item()
    return dict(max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                relative_l2=error/norm if norm else (0. if error == 0 else None),
                zero_reference=norm == 0, top1_reference=r.argmax(-1).item(),
                top1_candidate=c.argmax(-1).item(), reference_margin=margin(r),
                candidate_margin=margin(c))


def check_limits(limits):
    if not isinstance(limits, dict) or set(limits) != set(METRICS):
        raise ValueError('three numeric limits required')
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in limits.values()):
        raise ValueError('limits must be finite and nonnegative')
    return limits


def evaluate_metrics(metrics, limits):
    if limits is None:
        return 'diagnostic'
    check_limits(limits)
    if any(metrics[k] is None or not math.isfinite(metrics[k]) or metrics[k] > limits[k] for k in METRICS):
        return 'fail'
    if metrics['top1_reference'] != metrics['top1_candidate']:
        return 'needs_review'
    return 'pass'


def validate_tolerances(document, expected_fingerprint):
    if (document.get('approved') is not True
            or document.get('fingerprint') != expected_fingerprint
            or not isinstance(document.get('review_reason'), str)
            or not document['review_reason'].strip()):
        raise ValueError('unapproved tolerances or mismatched fingerprint')
    return check_limits(document.get('limits'))


def aggregate(rows):
    states = {r['status'] for r in rows}
    for status in ('fail', 'needs_review', 'diagnostic'):
        if status in states:
            return status
    return 'pass' if rows else 'diagnostic'


def state_metadata(model, cache, processed):
    if model is None:  # dependency injection used only by CPU verifier tests
        return None
    result = snapshot_cache(cache, model.config.text_config.layer_types, processed)
    if not result['complete']:
        raise ValueError('incomplete hybrid state')
    return result


def verify_prefixes(model, prompt_ids, continuation_ids, *, limits=None, step=forward_last):
    validate_ids(prompt_ids)
    validate_ids(continuation_ids)
    prefix, processed, cache = prompt_ids, 0, None
    rows = []
    reference = candidate = None
    try:
        for j in range(continuation_ids.shape[1] + 1):
            token = prompt_ids if j == 0 else continuation_ids[:, j-1:j]
            if j:
                prefix = torch.cat([prefix, token], dim=1)
            reference, _ = step(model, prefix, processed_tokens=0, cache=None, use_cache=False)
            candidate, cache = step(model, token, processed_tokens=processed, cache=cache, use_cache=True)
            if cache is None:
                raise ValueError('cached path did not return state')
            start = processed
            processed += token.shape[1]
            m = compare_logits(reference, candidate)
            rows.append(dict(phase='prefill' if j == 0 else 'decode',
                             prefix_ids=prefix.tolist()[0], candidate_input_ids=token.tolist()[0],
                             position_start=start, processed_tokens=processed,
                             metrics=m, status=evaluate_metrics(m, limits),
                             cache=state_metadata(model, cache, processed)))
    finally:
        cache = reference = candidate = None
    status = aggregate(rows)
    return dict(steps=rows, status=status, passed=status == 'pass')


def cached_sequence(model, prompt, continuation):
    """Return only CPU logits/JSON; fresh cache at position zero for each call."""
    cache, logits, processed = None, None, 0
    outputs, trace = [], []
    try:
        for j in range(continuation.shape[1] + 1):
            token = prompt if j == 0 else continuation[:, j-1:j]
            start = processed
            logits, cache = forward_last(model, token, processed_tokens=start, cache=cache, use_cache=True)
            processed += token.shape[1]
            outputs.append(logits.detach().float().cpu().clone())
            trace.append(dict(position_start=start, input_ids=token.tolist()[0],
                              cache=state_metadata(model, cache, processed)))
    finally:
        cache = logits = None
    return outputs, trace


def verify_isolation(model, prompt_a, prompt_b, continuation, eos, *, limits=None):
    def generation_capture(prompt):
        captured = []
        def capture(*a, **kw):
            logits, updated = forward_last(*a, **kw)
            captured.append((a[1].tolist()[0], logits.detach().float().cpu().clone()))
            return logits, updated
        result = run_greedy(model, prompt, eos, 4, step=capture)
        return result, captured

    baseline_generation, baseline_scores = generation_capture(prompt_b)
    baseline, baseline_trace = cached_sequence(model, prompt_b, continuation)
    rows = []
    fault = {}
    class InjectedFailure(RuntimeError):
        pass
    def injected_step(*a, **kw):
        logits, cache = forward_last(*a, **kw)
        if cache is None:
            raise ValueError('fault injection did not create state')
        if kw.get('cache') is None:
            return logits, cache  # Prefill must return; caller now owns state.
        fault.update(phase='decode', received_existing_cache=True,
                     position_start=kw['processed_tokens'], input_ids=a[1].tolist()[0],
                     cache_after_forward=state_metadata(model, cache, kw['processed_tokens'] + a[1].shape[1]))
        raise InjectedFailure('deliberate failure after stateful forward')
    for mode in ('after_success', 'after_exception'):
        injected = False
        a_result = None
        if mode == 'after_success':
            a_result = run_greedy(model, prompt_a, eos, 4, step=forward_last)
        else:
            try:
                run_greedy(model, prompt_a, eos, 4, step=injected_step)
            except InjectedFailure:
                injected = True
            if not injected:
                raise ValueError('expected injected exception not observed')
        generation, scores = generation_capture(prompt_b)
        actual, trace = cached_sequence(model, prompt_b, continuation)
        comparisons = []
        for ref, candidate in zip(baseline, actual):
            m = compare_logits(ref, candidate)
            comparisons.append(dict(metrics=m, status=evaluate_metrics(m, limits)))
        generation_steps = []
        if len(scores) != len(baseline_scores):
            generation_steps.append(dict(status='fail', reason='generation length differs'))
        prefix_same = True
        for (ref_ids, ref), (candidate_ids, candidate) in zip(baseline_scores, scores):
            prefix_same = prefix_same and ref_ids == candidate_ids
            if not prefix_same:
                generation_steps.append(dict(status='fail', reason='generation prefix diverged'))
                continue
            m = compare_logits(ref, candidate)
            generation_steps.append(dict(metrics=m, status=evaluate_metrics(m, limits)))
        if generation['output_ids'] != baseline_generation['output_ids'] or generation['stop_reason'] != baseline_generation['stop_reason']:
            generation_steps.append(dict(status='fail', reason='generation output or stop differs'))
        rows.append(dict(case=mode, status=aggregate(comparisons + generation_steps), steps=comparisons,
                         injected_exception_observed=injected, trace=trace, a_generation=a_result,
                         fault=dict(fault) if injected else None, generation=generation,
                         generation_steps=generation_steps))
    status = aggregate(rows)
    return dict(status=status, passed=status == 'pass', baseline_trace=baseline_trace,
                baseline_generation=baseline_generation, cases=rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root):
    import transformers
    return dict(revision='ed182e32090db791077e12e0f58d22f3daafa173', dtype='bf16',
                python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                backend='eager full attention / existing torch linear fallback',
                tf32=torch.backends.cuda.matmul.allow_tf32,
                files={**{name: sha(root/name) for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')},
                       'generation_config.json': sha(root/'generation_config.json') if (root/'generation_config.json').exists() else None},
                code={name: sha(Path(__file__).parent/name) for name in ('verify_generation.py', 'reference_runtime.py', 'minimal_generate.py', 'cache_inspection.py', 'inspect_model.py')})


def encode(tokenizer, text, device):
    return torch.tensor([inspect_prompt(tokenizer, text, thinking=False)['chat_ids']], dtype=torch.long, device=device)


def continuation_ids(tokenizer, device):
    ids = tokenizer.encode(CONTINUATION, add_special_tokens=False)
    if len(ids) < 4:
        raise ValueError('continuation must have at least four tokens')
    return torch.tensor([ids[:4]], dtype=torch.long, device=device)


def calibrate(root, device):
    model, tok, manifest, _ = load_reference(root, device, 'bf16')
    fp = fingerprint(root)
    prefixes, bf16, repeats, states = [], [], [], []
    cont = continuation_ids(tok, device)
    for text in CALIBRATION:
        prompt = encode(tok, text, device)
        for j in range(5):
            ids = torch.cat([prompt, cont[:, :j]], dim=1)
            a, _ = forward_last(model, ids, use_cache=False)
            b, _ = forward_last(model, ids, use_cache=False)
            repeats.append(compare_logits(a, b))
            bf16.append(a.detach().float().cpu().clone())
            prefixes.append(ids.tolist()[0])
    # Observe actual state dtypes for both precision settings, outside calibration math.
    _, cache = forward_last(model, prompt, use_cache=True)
    states.append(state_metadata(model, cache, prompt.shape[1]))
    del a, b, cache, model
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _, _ = load_reference(root, device, 'fp32')
    precision = []
    for ids, saved in zip(prefixes, bf16):
        out, _ = forward_last(model, torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
        precision.append(compare_logits(out.detach().float().cpu(), saved))
    _, cache = forward_last(model, prompt, use_cache=True)
    states.append(state_metadata(model, cache, prompt.shape[1]))
    del out, cache, model
    gc.collect()
    torch.cuda.empty_cache()
    rows = repeats + precision
    if any(r['relative_l2'] is None for r in rows):
        raise ValueError('cannot calibrate undefined relative error')
    floors = dict(max_abs=1e-4, mean_abs=1e-5, relative_l2=1e-5)
    limits = {k: max(floors[k], 4 * max(r[k] for r in rows)) for k in METRICS}
    evidence = dict(prefixes=prefixes, repeat_bf16=repeats, fp32_vs_bf16=precision)
    calibration_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    tolerance = dict(approved=False, review_reason='', fingerprint=fp, limits=limits,
                     calibration_sha256=calibration_hash,
                     maxima={k: dict(value=max(r[k] for r in rows), combined_row=max(range(len(rows)), key=lambda i: rows[i][k])) for k in METRICS})
    return dict(status='diagnostic', passed=False, model=manifest, fingerprint=fp,
                calibration=evidence, states_by_precision=states, tolerance_candidate=tolerance)


def verify(root, device, tolerance_path):
    model, tok, manifest, eos = load_reference(root, device, 'bf16')
    fp = fingerprint(root)
    limits = None
    if tolerance_path:
        limits = validate_tolerances(json.loads(tolerance_path.read_text()), fp)
    cont = continuation_ids(tok, device)
    cases = []
    for text in ACCEPTANCE:
        r = verify_prefixes(model, encode(tok, text, device), cont, limits=limits)
        r['prompt'] = text
        cases.append(r)
        print('PREFIX', text, r['status'], max(x['metrics']['max_abs'] for x in r['steps']), flush=True)
    ordinary = [x for x in tok.encode('hello', add_special_tokens=False) if x not in tok.all_special_ids]
    if not ordinary:
        raise ValueError('no ordinary token for T=1')
    r = verify_prefixes(model, torch.tensor([[ordinary[0]]], device=device), cont, limits=limits)
    r['prompt'] = 'T=1 ordinary raw token'
    cases.append(r)
    isolation = verify_isolation(model, encode(tok, ACCEPTANCE[0], device), encode(tok, '你好', device), cont, eos, limits=limits)
    status = aggregate(cases + [isolation])
    return dict(status=status, passed=status == 'pass', fingerprint=fp, model=manifest,
                cases=cases, isolation=isolation, limits=limits,
                tolerance_sha256=sha(tolerance_path) if tolerance_path else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--calibrate', action='store_true')
    mode.add_argument('--verify', action='store_true')
    p.add_argument('--model', type=Path, default=Path('/home/ubuntu/huggingface/Qwen3.5-4B'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tolerance-file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tolerance_output = args.output.with_name(args.output.stem + '-tolerances.json')
    if args.output.exists() or (args.calibrate and tolerance_output.exists()):
        p.error('refusing to overwrite results')
    if args.calibrate and args.tolerance_file:
        p.error('calibration cannot consume acceptance tolerances')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    result = calibrate(args.model, args.device) if args.calibrate else verify(args.model, args.device, args.tolerance_file)
    result['gpu_inventory'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    import os
    result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    if args.calibrate:
        with tolerance_output.open('x') as f:
            json.dump(result['tolerance_candidate'], f, ensure_ascii=False, indent=2, allow_nan=False)
    print('RESULT', args.output, result['status'], flush=True)
    return 1 if result['status'] in ('fail', 'needs_review') else 0


if __name__ == '__main__':
    raise SystemExit(main())
