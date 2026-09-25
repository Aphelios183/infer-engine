"""第一课：只读取本地配置和 tokenizer；不加载权重、不执行模型、不写文件。"""
import argparse
from collections.abc import Mapping
import hashlib
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


def inspect_prompt(tokenizer, prompt, thinking=False):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt 必须为非空文本')
    if type(thinking) is not bool:
        raise ValueError('thinking 必须为 bool')
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or 'enable_thinking' not in template:
        raise ValueError('chat template 未声明 enable_thinking；不能声称已关闭 thinking')
    messages = [{'role': 'user', 'content': prompt}]
    kwargs = dict(add_generation_prompt=True, enable_thinking=thinking)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    other = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=not thinking)
    if rendered == other:
        raise ValueError('thinking 开关未改变模板输出；本课不能确认该开关有效')
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
                template_settings=kwargs, thinking_switch_verified=True,
                template_sha256=hashlib.sha256(template.encode('utf-8')).hexdigest(),
                rendered_sha256=hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
                eos_token=tokenizer.eos_token, eos_token_id=tokenizer.eos_token_id,
                tokenizer_base_vocab_size=tokenizer.vocab_size,
                tokenizer_length=len(tokenizer),
                tokens=tokenizer.convert_ids_to_tokens(ids))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--model', type=Path, help='本地模型目录；默认 Qwen3.5-4B')
    sources.add_argument('--config-file', type=Path, help='只读 JSON 文件，仅配合 --config-only')
    parser.add_argument('--config-only', action='store_true', help='只读配置；不导入 torch/tokenizer')
    parser.add_argument('--thinking', choices=('on', 'off'), default='off')
    parser.add_argument('--prompt', default='用一句话解释 KV Cache。')
    args = parser.parse_args(argv)
    if args.config_file is not None and not args.config_only:
        parser.error('--config-file 必须配合 --config-only；教学样例不能用于分词或加载模型')
    model_path = args.model or Path('/home/ubuntu/huggingface/Qwen3.5-4B')
    config_path = args.config_file if args.config_file is not None else model_path / 'config.json'
    try:
        raw = config_path.read_bytes()
        config = json.loads(raw)
        report = describe_config(config)
    except (OSError, ValueError) as exc:
        parser.error(f'本地配置读取失败: {config_path}: {exc}。不会自动下载或退回旧模型。')
    report.update(config_source=str(config_path),
                  config_sha256=hashlib.sha256(raw).hexdigest(),
                  is_teaching_fixture='_fixture_note' in config,
                  fixture_note=config.get('_fixture_note'))
    if args.config_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if '_fixture_note' in config:
        parser.error('教学缩减样例只支持 --config-only，不是可运行模型目录')
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                 trust_remote_code=False)
        result = inspect_prompt(tokenizer, args.prompt, thinking=args.thinking == 'on')
    except (OSError, ValueError) as exc:
        parser.error(f'本地 tokenizer/模板检查失败: {exc}；本脚本不会下载资产')
    print('=== 模型配置与 Full Attention 理论 KV（未分配张量） ===')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('\n=== 文本到 token ===')
    print('原始文本:', repr(result['prompt']))
    print('模板后文本:', repr(result['rendered']))
    print('原始 token 数:', result['raw_length'], '模型输入 token 数:', result['chat_length'])
    print('输入 IDs:', result['chat_ids'])
    print('EOS:', result['eos_token'], result['eos_token_id'])
    print('模板设置:', result['template_settings'])
    print('模板 SHA256:', result['template_sha256'])
    print('thinking 开关已改变模板；尚未运行模型，不推断模型生成行为。')
    print('tokenizer 基础词表 / 含 added tokens:',
          result['tokenizer_base_vocab_size'], result['tokenizer_length'])
    print('逐位置 token 表示（字节级分词标记不一定是可读汉字）:')
    for position, (token_id, token) in enumerate(zip(result['chat_ids'], result['tokens'])):
        print(f'{position:3d}  {token_id:6d}  {token!r}')
    print('模板两种编码一致；完整序列往返解码一致。')
    print('今天只到 token IDs；没有加载权重、生成 logits 或生成回答。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
