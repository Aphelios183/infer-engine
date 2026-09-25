"""第一课：只读取本地配置和 tokenizer；不加载权重、不执行模型、不写文件。"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} 必须为正整数（不能是 bool）')
    return value


def describe_config(config, batch=1, length=4096, bytes_per_element=2):
    """只算 Full Attention 的理论 K/V；不分配张量、不猜线性状态大小。"""
    if not isinstance(config, dict):
        raise ValueError('model_type 必须来自配置对象')
    kind = config.get('model_type')
    if kind == 'qwen3_5':
        text = config.get('text_config')
        if not isinstance(text, dict) or text.get('model_type') != 'qwen3_5_text':
            raise ValueError('text_config.model_type 必须为 qwen3_5_text')
    elif kind == 'qwen3':
        text = config
    else:
        raise ValueError(f'不支持 model_type={kind!r}')

    _positive_int(batch, 'batch')
    _positive_int(length, 'length')
    _positive_int(bytes_per_element, 'bytes_per_element')
    for field in ('num_hidden_layers', 'hidden_size', 'num_attention_heads',
                  'num_key_value_heads', 'head_dim', 'vocab_size'):
        _positive_int(text.get(field), field)
    layers = text['num_hidden_layers']
    q_heads = text['num_attention_heads']
    kv_heads = text['num_key_value_heads']
    head_dim = text['head_dim']
    if q_heads % kv_heads:
        raise ValueError('num_attention_heads 必须能被 num_key_value_heads 整除')

    if kind == 'qwen3_5':
        layer_types = text.get('layer_types')
        if (not isinstance(layer_types, list) or len(layer_types) != layers
                or any(not isinstance(t, str) or t not in
                       ('full_attention', 'linear_attention') for t in layer_types)):
            raise ValueError('layer_types 必须与层数一致，且只含 full_attention/linear_attention')
        layer_types = list(layer_types)
    else:
        layer_types = ['full_attention'] * layers
    full_indices = [i for i, t in enumerate(layer_types) if t == 'full_attention']
    linear_indices = [i for i, t in enumerate(layer_types) if t == 'linear_attention']
    linear_config = {}
    if linear_indices:
        for field in ('linear_num_key_heads', 'linear_num_value_heads',
                      'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim'):
            linear_config[field] = _positive_int(text.get(field), field)
        linear_config['configured_state_dtype'] = text.get('mamba_ssm_dtype')

    kv_bytes = 2 * len(full_indices) * batch * length * kv_heads * head_dim * bytes_per_element
    return dict(model_type=kind, text_model_type=text.get('model_type'),
                layers=layers, hidden_size=text['hidden_size'],
                q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
                q_heads_per_kv=q_heads // kv_heads,
                q_projection_width=q_heads * head_dim,
                q_projection_scope='query_features_only',
                kv_projection_width=kv_heads * head_dim,
                model_vocab_size=text['vocab_size'],
                layer_types=layer_types,
                full_attention_layers=len(full_indices),
                linear_attention_layers=len(linear_indices),
                full_attention_layer_indices=full_indices,
                linear_attention_layer_indices=linear_indices,
                kv_scope='full_attention_only', linear_state_bytes=None,
                linear_state_status='未测：未加载模型或创建状态张量',
                linear_config=linear_config,
                configured_weight_dtype=text.get('dtype', text.get('torch_dtype')),
                bytes_per_element=bytes_per_element,
                assumed_kv_dtype=f'假设每元素 {bytes_per_element} 字节；不是实测 dtype',
                batch=batch, length=length, kv_bytes=kv_bytes,
                kv_mib=kv_bytes / 1024**2)


def inspect_prompt(tokenizer, prompt):
    messages = [{'role': 'user', 'content': prompt}]
    kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    # 模板已包含特殊标记，后续不重复自动添加 special tokens。
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    direct = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    # 当前 Transformers 5.8 返回 BatchEncoding；也接受旧版的 flat list。
    direct_ids = direct['input_ids'] if isinstance(direct, Mapping) else direct
    if ids != direct_ids:
        raise AssertionError('模板直接编码与手动编码不一致')
    decoded = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded != rendered:
        raise AssertionError('模板文本与完整 token 序列往返解码不一致')
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return dict(prompt=prompt, raw_ids=raw_ids, raw_length=len(raw_ids),
                rendered=rendered, chat_ids=ids, chat_length=len(ids),
                template_encoding_matches=True, roundtrip_matches=True,
                eos_token=tokenizer.eos_token, eos_token_id=tokenizer.eos_token_id,
                tokenizer_base_vocab_size=tokenizer.vocab_size,
                tokenizer_length=len(tokenizer),
                tokens=tokenizer.convert_ids_to_tokens(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/ubuntu/huggingface/Qwen3-0.6B'))
    parser.add_argument('--prompt', default='用一句话解释 KV Cache。')
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error('model 必须是已存在的本地目录；不会从网络下载')
    config = json.loads((args.model / 'config.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True,
                                              trust_remote_code=False)
    print('=== 模型配置与理论 KV 大小（未分配 KV） ===')
    print(json.dumps(describe_config(config), ensure_ascii=False, indent=2))
    result = inspect_prompt(tokenizer, args.prompt)
    print('\n=== 文本到 token ===')
    print('原始文本:', repr(result['prompt']))
    print('模板后文本:', repr(result['rendered']))
    print('原始 token 数:', result['raw_length'], '模型输入 token 数:', result['chat_length'])
    print('输入 IDs:', result['chat_ids'])
    print('EOS:', result['eos_token'], result['eos_token_id'])
    print('tokenizer 基础词表 / 含 added tokens:',
          result['tokenizer_base_vocab_size'], result['tokenizer_length'])
    print('逐位置 token 表示（字节级分词标记不一定是可读汉字）:')
    for position, (token_id, token) in enumerate(zip(result['chat_ids'], result['tokens'])):
        print(f'{position:3d}  {token_id:6d}  {token!r}')
    print('模板两种编码一致；完整序列往返解码一致。')
    print('今天只到 token IDs；没有加载权重、生成 logits 或生成回答。')


if __name__ == '__main__':
    main()
