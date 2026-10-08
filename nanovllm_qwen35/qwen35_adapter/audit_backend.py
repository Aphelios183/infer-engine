"""Read checkpoint headers and installed reference source; no model/GPU load."""
import argparse
import hashlib
import importlib.util
import json
import struct
from collections import Counter
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def audit(model_dir):
    root = Path(model_dir).resolve()
    config = json.loads((root / 'config.json').read_text())
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    headers = {}
    for shard in sorted(set(weights.values())):
        path = (root / shard).resolve()
        if path.parent != root:
            raise ValueError('shard outside model directory')
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError('invalid header size')
            headers[shard] = json.loads(f.read(length))
    tensors = {k: headers[v][k] for k, v in weights.items()}
    for name, meta in tensors.items():
        start, end = meta['data_offsets']
        if start < 0 or end < start:
            raise ValueError(f'invalid tensor offsets: {name}')
    prefix = 'model.language_model.'
    selected = {k: {'shape': v['shape'], 'dtype': v['dtype']} for k, v in tensors.items()
                if k.startswith(prefix) and ('.layers.0.' in k or '.layers.3.' in k
                   or '.embed_tokens.' in k or k == prefix + 'norm.weight')}
    tf = Path(importlib.util.find_spec('transformers').origin).parent
    sources = [tf / 'models/qwen3_5/modeling_qwen3_5.py', tf / 'cache_utils.py']
    packages = {}
    for name in ['torch', 'transformers', 'safetensors', 'flash-linear-attention', 'causal-conv1d']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        'scope': 'headers/source metadata only; no tensor payload or model forward',
        'packages': packages,
        'config_sha256': hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'tensor_count': len(tensors),
        'groups': dict(Counter('text' if k.startswith(prefix) else
                               'vision' if k.startswith('model.visual.') else
                               'mtp' if k.startswith('mtp.') else 'other' for k in tensors)),
        'dtypes': dict(Counter(v['dtype'] for v in tensors.values())),
        'lm_head_keys': [k for k in tensors if 'lm_head' in k],
        'tie_word_embeddings': config['text_config']['tie_word_embeddings'],
        'selected_weights': selected,
    }


def probe_cache_dtype():
    """Exercise actual installed cache class with tiny CPU tensors, not a model."""
    import torch
    from transformers.cache_utils import LinearAttentionLayer
    cache = LinearAttentionLayer()
    cache.update_conv_state(torch.zeros(1, 2, 4, dtype=torch.bfloat16))
    source = torch.full((1, 1, 2, 2), 1.001, dtype=torch.float32)
    stored = cache.update_recurrent_state(source)
    return {'conv': str(cache.conv_states.dtype), 'incoming_recurrent': str(source.dtype),
            'stored_recurrent': str(stored.dtype), 'input_value': source.flatten()[0].item(),
            'stored_value': stored.flatten()[0].item(),
            'scope': 'isolated CPU cache class; not full-model runtime dtype verification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--probe-cache', action='store_true')
    args = parser.parse_args()
    result = audit(args.model_dir)
    if args.probe_cache:
        result['cache_probe'] = probe_cache_dtype()
    print(json.dumps(result, indent=2))
