"""Strict text-only checkpoint mapping PLAN. Never loads tensor payloads/model."""
import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

from .contracts import parse_text_contract


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def read_headers(root):
    """Check index/header bijection, tensor byte ranges and supported file dtypes."""
    root = Path(root).resolve()
    index = read_json(root / 'model.safetensors.index.json')
    mapping = index['weight_map']
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('empty or invalid weight_map')
    all_tensors = {}
    total_bytes = 0
    for filename in sorted(set(mapping.values())):
        path = (root / filename).resolve()
        if path.parent != root or path.suffix != '.safetensors':
            raise ValueError('shard must be a local safetensors file')
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('truncated shard header')
            size = struct.unpack('<Q', raw)[0]
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError('invalid header size')
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError('truncated shard header')
            header = json.loads(raw, object_pairs_hook=unique_object)
        payload_size = path.stat().st_size - 8 - size
        ranges = []
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            if name in all_tensors:
                raise ValueError(f'duplicate tensor across shards: {name}')
            if mapping.get(name) != filename:
                raise ValueError(f'index/header mismatch: {name}')
            shape, dtype, offsets = meta['shape'], meta['dtype'], meta['data_offsets']
            if (not isinstance(shape, list) or any(type(x) is not int or x <= 0 for x in shape)
                    or dtype not in ('BF16', 'F16', 'F32')):
                raise ValueError(f'unsupported shape/dtype: {name}')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int for x in offsets)):
                raise ValueError(f'invalid offsets: {name}')
            start, end = offsets
            expected_bytes = math.prod(shape) * (4 if dtype == 'F32' else 2)
            if not 0 <= start <= end <= payload_size or end - start != expected_bytes:
                raise ValueError(f'invalid tensor byte range: {name}')
            ranges.append((start, end))
            total_bytes += end - start
            all_tensors[name] = dict(meta, shard=filename)
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError('overlap or gap in shard payload')
            cursor = end
        if cursor != payload_size:
            raise ValueError('unaccounted shard payload')
    if set(mapping) != set(all_tensors):
        raise ValueError('index keys missing from headers')
    if index.get('metadata', {}).get('total_size', total_bytes) != total_bytes:
        raise ValueError('total_size mismatch')
    return all_tensors


def expected_text(config):
    parse_text_contract(config)
    t = config['text_config']
    for name in ('vocab_size', 'intermediate_size'):
        if type(t.get(name)) is not int or t[name] <= 0:
            raise ValueError(f'invalid {name}')
    if (t.get('tie_word_embeddings') is not True or t.get('attention_bias') is not False
            or t.get('attn_output_gate') is not True or t.get('hidden_act') != 'silu'):
        raise ValueError('plan only supports tied, bias-free, gated dense SiLU text model')
    h, inter = t['hidden_size'], t['intermediate_size']
    expected = {}

    def add(name, shape, semantics='copy_preserve_layout'):
        expected['model.language_model.' + name] = {
            'target': 'model.' + name, 'shape': shape, 'semantics': semantics}

    add('embed_tokens.weight', [t['vocab_size'], h])
    add('norm.weight', [h], 'rmsnorm_one_plus_weight')
    for i, kind in enumerate(t['layer_types']):
        p = f'layers.{i}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            add(p + name + '.weight', [h], 'rmsnorm_one_plus_weight')
        for name in ('gate_proj', 'up_proj'):
            add(p + 'mlp.' + name + '.weight', [inter, h])
        add(p + 'mlp.down_proj.weight', [h, inter])
        if kind == 'full_attention':
            q = t['num_attention_heads'] * t['head_dim']
            kv = t['num_key_value_heads'] * t['head_dim']
            add(p + 'self_attn.q_proj.weight', [2 * q, h], 'per_head_Q_then_gate')
            for name in ('k_proj', 'v_proj'):
                add(p + 'self_attn.' + name + '.weight', [kv, h])
            add(p + 'self_attn.o_proj.weight', [h, q])
            for name in ('q_norm', 'k_norm'):
                add(p + 'self_attn.' + name + '.weight', [t['head_dim']], 'rmsnorm_one_plus_weight')
        else:
            k = t['linear_num_key_heads'] * t['linear_key_head_dim']
            v = t['linear_num_value_heads'] * t['linear_value_head_dim']
            n = t['linear_num_value_heads']
            add(p + 'linear_attn.in_proj_qkv.weight', [2*k + v, h], 'Q_K_V_widths:' + str([k,k,v]))
            add(p + 'linear_attn.in_proj_z.weight', [v, h])
            for name in ('a', 'b'):
                add(p + f'linear_attn.in_proj_{name}.weight', [n, h])
            add(p + 'linear_attn.conv1d.weight', [2*k + v, 1, t['linear_conv_kernel_dim']])
            add(p + 'linear_attn.out_proj.weight', [h, v])
            add(p + 'linear_attn.norm.weight', [t['linear_value_head_dim']], 'gated_norm_direct_weight')
            add(p + 'linear_attn.A_log', [n])
            add(p + 'linear_attn.dt_bias', [n])
    return expected


def build_plan(config, tensors):
    expected = expected_text(config)
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(k for k in tensors if k not in expected
                        and not k.startswith(('model.visual.', 'mtp.')))
    if missing or unexpected:
        raise ValueError(f'missing={missing}; unexpected={unexpected}')
    entries = []
    for source, spec in expected.items():
        meta = tensors[source]
        if meta['shape'] != spec['shape']:
            raise ValueError(f'shape mismatch: {source}: {meta["shape"]} != {spec["shape"]}')
        if meta['dtype'] not in ('BF16', 'F16', 'F32'):
            raise ValueError(f'unsupported text dtype: {source}')
        entries.append(dict(spec, source=source, dtype=meta['dtype'], shard=meta.get('shard')))
    if len({e['target'] for e in entries}) != len(entries):
        raise ValueError('duplicate target mapping')
    ignored = {group: sorted(k for k in tensors if k.startswith(prefix))
               for group, prefix in [('vision', 'model.visual.'), ('mtp', 'mtp.')]}
    return {'status': 'metadata_plan_passed_not_loaded', 'mapped_count': len(entries),
            'ignored_counts': {k: len(v) for k,v in ignored.items()},
            'dtype_counts': dict(Counter(e['dtype'] for e in entries)),
            'aliases': {'lm_head.weight': 'model.embed_tokens.weight'},
            'entries': entries, 'ignored': ignored}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    result = build_plan(read_json(Path(args.model_dir) / 'config.json'), read_headers(args.model_dir))
    print(json.dumps(result if args.full else {k:v for k,v in result.items()
                                             if k not in ('entries', 'ignored')}, indent=2))
