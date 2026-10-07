"""Lesson 4: validate configuration before allocating any model state."""

import json
from pathlib import Path


def build_layer_maps(layer_types):
    """Student implementation: original layer ID -> compact pool layer index."""
    full_layer_to_kv = {}
    linear_layer_to_state = {}
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == 'full_attention':
            full_layer_to_kv[layer_idx] = len(full_layer_to_kv)
        elif layer_type == 'linear_attention':
            linear_layer_to_state[layer_idx] = len(linear_layer_to_state)
        else:
            raise ValueError(
                f'未知层类型 {layer_type!r} 出现在 layer {layer_idx}；'
                "只接受 'full_attention' 或 'linear_attention'"
            )
    return full_layer_to_kv, linear_layer_to_state


def parse_text_contract(config):
    """Accept only the dense multimodal wrapper config used by this course.

    Presence of vision_config does not mean vision execution is supported.
    Shapes below describe a proposed pool layout, not allocated tensors.
    """
    if not isinstance(config, dict) or config.get('model_type') != 'qwen3_5':
        raise ValueError('expected dense qwen3_5 wrapper config')
    text = config.get('text_config')
    if not isinstance(text, dict) or text.get('model_type') != 'qwen3_5_text':
        raise ValueError('expected qwen3_5_text text_config')
    fields = ('num_hidden_layers', 'hidden_size', 'num_attention_heads',
              'num_key_value_heads', 'head_dim', 'max_position_embeddings',
              'linear_num_key_heads', 'linear_num_value_heads',
              'linear_key_head_dim', 'linear_value_head_dim',
              'linear_conv_kernel_dim')
    for name in fields:
        value = text.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    layer_types = text.get('layer_types')
    if not isinstance(layer_types, list):
        raise ValueError('layer_types must be an explicit list')
    if len(layer_types) != text['num_hidden_layers']:
        raise ValueError('layer_types length differs from num_hidden_layers')
    if text['num_attention_heads'] % text['num_key_value_heads']:
        raise ValueError('attention heads must be divisible by KV heads')
    if text['linear_num_value_heads'] % text['linear_num_key_heads']:
        raise ValueError('linear value heads must be divisible by key heads')
    full, linear = build_layer_maps(layer_types)
    channels = (2 * text['linear_num_key_heads'] * text['linear_key_head_dim']
                + text['linear_num_value_heads'] * text['linear_value_head_dim'])
    return {
        'model_type': config['model_type'],
        'num_hidden_layers': text['num_hidden_layers'],
        'hidden_size': text['hidden_size'],
        'full_layer_to_kv': full,
        'linear_layer_to_state': linear,
        'full_kv_heads': text['num_key_value_heads'],
        'full_head_dim': text['head_dim'],
        'conv_shape_per_layer_per_request': [channels, text['linear_conv_kernel_dim']],
        'recurrent_shape_per_layer_per_request': [
            text['linear_num_value_heads'], text['linear_key_head_dim'],
            text['linear_value_head_dim']],
        'declared_model_dtype': text.get('dtype', text.get('torch_dtype')),
        'declared_ssm_dtype': text.get('mamba_ssm_dtype'),
        'runtime_state_dtype': 'UNVERIFIED: inspect actual backend tensors',
        'eos_token_id': text.get('eos_token_id'),
        'native_inference_ready': False,
    }


def load_contract(config_path):
    return parse_text_contract(json.loads(Path(config_path).read_text(encoding='utf-8')))


def guard_unimplemented_model(config):
    """Called by copied nano Config before AutoConfig or GPU initialization."""
    model_types = [config.get('model_type'), config.get('text_config', {}).get('model_type')]
    if any(isinstance(t, str) and t.startswith('qwen3_5') for t in model_types):
        raise NotImplementedError(
            'Qwen3.5 native execution is not implemented. '
            'Use python -m qwen35_adapter to inspect its configuration; '
            'do not route it through Qwen3ForCausalLM.'
        )
