"""Strict CPU loader into a fresh parameter skeleton; no forward/GPU backend."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from .parameters import Qwen35TextParameters
from .weight_plan import read_json, read_headers, build_plan


DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}


def load_text_parameters(model_dir, *, verify_values=False):
    """Return only a completely loaded fresh object. Never mutate a live model.

    CPU copies own their storage after closing safetensors handles. Preserve each
    checkpoint dtype. On failure no usable model is returned; caller must not retry
    using partially loaded internals. Keep checkpoint files immutable during loading.
    """
    root = Path(model_dir).resolve()
    config = read_json(root / 'config.json')
    plan = build_plan(config, read_headers(root))
    model = Qwen35TextParameters(config)
    expected = {e['target'] for e in plan['entries']}
    if set(dict(model.named_parameters())) != expected:
        raise ValueError('model parameter names do not match loading plan')
    shards = defaultdict(list)
    for entry in plan['entries']:
        shards[entry['shard']].append(entry)
    loaded = set()
    verified = 0
    with torch.no_grad():
        for shard, entries in sorted(shards.items()):
            with safe_open(str(root / shard), framework='pt', device='cpu') as handle:
                for entry in entries:
                    source = handle.get_tensor(entry['source'])
                    if list(source.shape) != entry['shape'] or source.dtype != DTYPES[entry['dtype']]:
                        raise ValueError(f'payload shape/dtype mismatch: {entry["source"]}')
                    target = entry['target']
                    if target in loaded:
                        raise ValueError(f'duplicate parameter load: {target}')
                    model.replace_parameter(target, source.clone(memory_format=torch.contiguous_format))
                    if verify_values:
                        # Compare all bytes, including NaN payloads; NOT a finite-value/model correctness test.
                        actual = model.get_parameter(target).detach()
                        if not torch.equal(actual.view(torch.uint8), source.view(torch.uint8)):
                            raise ValueError(f'byte comparison failed: {target}')
                        verified += 1
                    loaded.add(target)
                    del source
    model.tie_weights()  # embedding Parameter was replaced; rebind the head explicitly
    if loaded != expected:
        raise ValueError('incomplete loading')
    if any(p.is_meta or p.device.type != 'cpu' or p.requires_grad for p in model.parameters()):
        raise ValueError('unmaterialized or invalid parameter')
    all_names = set(dict(model.named_parameters(remove_duplicate=False)))
    if all_names != expected | set(plan['aliases']):
        raise ValueError('unexpected parameter or alias')
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('head embedding alias is broken')
    model.loaded = True
    model.eval()
    report = {
        'status': 'cpu_parameters_loaded_forward_not_implemented',
        'loaded_count': len(loaded), 'verified_tensors': verified,
        'unique_parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
        'dtype_counts': plan['dtype_counts'], 'ignored_counts': plan['ignored_counts'],
        'head_is_embedding_parameter': True,
        'head_storage_matches': model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
    }
    return model, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--verify-values', action='store_true')
    args = parser.parse_args()
    # Limit host parallelism for this one-off verification command, not library callers.
    torch.set_num_threads(2)
    model, report = load_text_parameters(args.model_dir, verify_values=args.verify_values)
    print(json.dumps(report, indent=2))
