### Task 1：配置模型类型分派与正确的 KV 算术

**Files:** Modify `week3/inspect_model.py`、`week3/test_inspect_model.py`；Create `week3/fixtures/qwen35_4b_config_minimal.json`。

**Interfaces:**

- Consumes：JSON dict，仅接受显式 `model_type=qwen3` 或 `qwen3_5`；后者要求 `text_config.model_type=qwen3_5_text`。
- Produces：`describe_config(config: dict, batch: int = 1, length: int = 4096, bytes_per_element: int = 2) -> dict`。保留旧 `kv_bytes`/`kv_mib` 字段但标注为 Full Attention K/V；新增 `full_attention_layers`、`linear_attention_layers`、`layer_types`、`linear_state_bytes=None`、`kv_scope='full_attention_only'`。
- `q_projection_width` 仅表示 Q 内容特征宽度，不含 output gate；新增 `q_projection_scope='query_features_only'`。

- [ ] **Step 1：添加配置样例和失败测试。** 缩减 fixture 按已确认配置填写 `model_type`、`text_config` 和本文设计第 4 节字段；`layer_types` 是 `[linear_attention,linear_attention,linear_attention,full_attention]` 重复 8 次，禁止省略后默认为 32 层普通 KV。

```python
import copy
import json
from pathlib import Path
from week3.inspect_model import describe_config

def test_hybrid_config(self):
    path = Path(__file__).parent / 'fixtures/qwen35_4b_config_minimal.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    r = describe_config(cfg)
    self.assertEqual((r['full_attention_layers'], r['linear_attention_layers']), (8, 24))
    self.assertEqual(r['kv_bytes'], 134217728)
    self.assertEqual(r['q_projection_width'], 4096)
    self.assertIsNone(r['linear_state_bytes'])
    self.assertEqual(describe_config(cfg, batch=2)['kv_bytes'], 268435456)
    for bad in ({'model_type': 'unknown'}, {'model_type': 'qwen3_5'}):
        with self.assertRaises(ValueError):
            describe_config(bad)
    broken = copy.deepcopy(cfg)
    broken['text_config']['layer_types'].pop()
    with self.assertRaisesRegex(ValueError, 'layer_types'):
        describe_config(broken)
    for b, t in ((0, 1), (True, 1), (1, -1), (1, 1.5)):
        with self.assertRaises(ValueError):
            describe_config(cfg, batch=b, length=t)
```

同时保留 Qwen3 的 448 MiB 回归：旧测试样例显式加 `model_type='qwen3'`，不能借机移除旧公式测试。增加非法层类型、缺失 head_dim、Q/KV 不整除、非法 bytes_per_element 用例。

- [ ] **Step 2：运行失败测试。** `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_inspect_model -v`。预期失败在未实现嵌套配置/分类字段，不能因为 fixture 路径错误而算作正确的 red。
- [ ] **Step 3：实现纯 Python 配置分派。** 验证后按以下核心逻辑计算；导入模块时不导入 torch/transformers、不初始化 CUDA。

```python
kind = config.get('model_type')
if kind == 'qwen3_5':
    text = config.get('text_config')
    if not isinstance(text, dict) or text.get('model_type') != 'qwen3_5_text':
        raise ValueError('text_config.model_type 必须是 qwen3_5_text')
    layer_types = text.get('layer_types')
elif kind == 'qwen3':
    text = config
    layer_types = ['full_attention'] * text['num_hidden_layers']
else:
    raise ValueError(f'不支持 model_type={kind!r}')
# 先逐字段检查存在、整数类型（排除 bool）、正值和层列表长度，再计算。
full_layers = layer_types.count('full_attention')
kv_bytes = 2 * full_layers * batch * length * text['num_key_value_heads'] * text['head_dim'] * bytes_per_element
```

缺字段统一转换为带字段名的 `ValueError`；错误不能默认为 0。输出 dtype 为明确假设，实际线性状态占用保持未测。
- [ ] **Step 4：重跑单元测试并核查 fixture 来源说明。** 不依赖本地大模型、GPU 或网络。
- [ ] **Step 5：审查本任务 diff 并形成独立提交。** 建议消息 `feat(week3): inspect Qwen3.5 hybrid text config`。学习检查：让用户解释为什么只乘 8 而不是 32。

