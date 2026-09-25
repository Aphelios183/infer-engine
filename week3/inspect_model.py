"""第一课：只读取本地配置和 tokenizer；不加载权重、不执行模型、不写文件。"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path


def describe_config(config, batch=1, length=4096):
    # 本课按 Qwen3 的显式 head_dim 计算，不套用 hidden_size / Q头数。
    layers = config['num_hidden_layers']
    q_heads = config['num_attention_heads']
    kv_heads = config['num_key_value_heads']
    head_dim = config['head_dim']
    if any(type(v) is not int or v <= 0
           for v in (batch, length, layers, q_heads, kv_heads, head_dim)):
        raise ValueError('batch、length 和模型维度必须为正整数')
    if q_heads % kv_heads:
        raise ValueError('本课要求 Q 头数能被 KV 头数整除')
    kv_bytes = 2 * layers * batch * length * kv_heads * head_dim * 2
    return dict(layers=layers, hidden_size=config['hidden_size'],
                q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
                q_heads_per_kv=q_heads // kv_heads,
                q_projection_width=q_heads * head_dim,
                kv_projection_width=kv_heads * head_dim,
                model_vocab_size=config['vocab_size'],
                assumed_kv_dtype='BF16/FP16: 2 bytes per element',
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
