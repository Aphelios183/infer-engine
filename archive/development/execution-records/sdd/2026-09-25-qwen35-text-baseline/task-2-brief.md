### Task 2：离线 CLI、模板校验与第一课迁移

**Files:** Modify `week3/inspect_model.py`、`week3/test_inspect_model.py`、`ROADMAP.md`、`week3/PLAN.md`、`week3/LESSON_01.md`、`week3/ENVIRONMENT.md`；Create `week3/test_tokenizer_integration.py`、`week3/HISTORICAL_QWEN3_06B.md`。

**Interfaces:**

- Consumes：Task 1 的 `describe_config`。
- Produces：`inspect_prompt(tokenizer, prompt: str, thinking: bool = False) -> dict`，保留旧返回字段，新增实际模板设置和 SHA256。`main(argv: list[str] | None = None) -> int` 支持 `--model`、`--config-only`、`--prompt`、`--thinking {on,off}`；默认模型为约定的 Qwen3.5 路径。

- [ ] **Step 1：先写失败测试。** 使用临时目录只放配置，验证 `--config-only` 不需要 tokenizer；空文本/空白文本拒绝。用 `unittest.mock` 的 tokenizer 替身模拟 Mapping 和 flat-list 两种返回，模拟两条编码路径不一致。

```python
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock
from week3.inspect_model import main, inspect_prompt

def test_config_only(self):
    fixture = Path(__file__).parent / 'fixtures/qwen35_4b_config_minimal.json'
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / 'config.json').write_bytes(fixture.read_bytes())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--model', folder, '--config-only']), 0)
        self.assertIn('full_attention', out.getvalue())

def test_empty_prompt_is_rejected(self):
    with self.assertRaises(ValueError):
        inspect_prompt(Mock(), '  ')
```

将旧 tokenizer 的固定 18 IDs 回归移到独立 integration 文件并标注 Qwen3-0.6B；默认无资产 CPU 测试不能因缺大模型失败。集成测试要求显式 `RUN_TOKENIZER_INTEGRATION=1`，未开启则报告 SKIP；开启后资产缺失必须 FAIL。
- [ ] **Step 2：运行 Task 2 的失败用例。** 沿用 Task 1 unittest 命令，检查失败与新 CLI/验证行为有关。
- [ ] **Step 3：实现离线入口和模板观察。** 配置模式在导入 AutoTokenizer 前返回；所有 tokenizer 加载固定 `local_files_only=True, trust_remote_code=False`。

```python
if args.config_only:
    print(json.dumps(describe_config(config), ensure_ascii=False, indent=2))
    return 0
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
messages = [{'role': 'user', 'content': args.prompt}]
rendered = tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=args.thinking == 'on')
ids = tokenizer.encode(rendered, add_special_tokens=False)
```

直接模板编码必须与 `ids` 一致；保存特殊 token、实际模板和完整解码结果。若模板没有可确认的 thinking 开关行为，明确拒绝声称已关闭；用 on/off 渲染及模板源验证，不能只因为传了参数就认定有效。将 prompt 和模板中的 image/video 特殊占位符留给生成入口明确拒绝，普通汉字“图片”不属于占位符。
- [ ] **Step 4：迁移第一课及计划。** 原实验结果完整保留到历史文件；新课不复用“18 tokens、1024 hidden、151645 EOS”。用 `[1,L] → [1,L,2560]` 讲形状；128 MiB 标注为理论 Full Attention KV。ENVIRONMENT 分为历史实测、当前目标、待验证；ROADMAP 保留 W2 完成状态并将 C++、压缩等标为非必选。
- [ ] **Step 5：CPU 回归、手动阅读文档命令、提交。** 建议消息 `docs(week3): migrate the first lesson to Qwen3.5`。此处教学停靠：用户先回答两类缓存与维度问题，再继续真正加载模型。

