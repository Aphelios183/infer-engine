# Qwen3.5-4B Text Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Week 3 迁移到 Qwen3.5-4B，完成自己的单请求生成控制、混合缓存观察和可复现的同模型正确性基线。

**Architecture:** 先用纯 CPU 配置/模板测试建立输入边界，再准备固定 revision 的官方资产。Transformers 负责参考模型计算，infer-engine 负责 Prefill/Decode、位置、停止和结果记录；生成和对拍共享一个轻量 forward 适配器，不实现新模型算子。

**Tech Stack:** Python 3.11.15、标准库 unittest、PyTorch 2.5.1+cu124、Transformers 5.8.0、现有 huggingface_hub、单张 NVIDIA L40。版本为已观察环境，不是建议重新安装的清单。

**Spec:** [已获用户确认的设计](../specs/2026-09-25-qwen35-text-baseline-design.md)。用户于 2026-09-25 回复“设计通过”；设计文件中的“等待审阅”为创建时的历史状态，以此审批记录为准。

## Global Constraints

- 项目主目录：`/home/ubuntu/infer-engine`。
- 模型目标改为 `Qwen/Qwen3.5-4B`；先单卡、纯文本、正确性，再扩展混合状态管理、多请求和一项有证据的性能改进。
- 环境解释器为 `/home/ubuntu/enter/envs/nanovllm/bin/python`。
- 计划模型目录为 `/home/ubuntu/huggingface/Qwen3.5-4B`；路径不存在不等于文件已准备。
- 起步使用 BF16、eval、inference_mode，关闭主动启用的 compile/CUDA Graph。加载采用 `trust_remote_code=False`。
- Week 1–2 不重做、不清理或覆盖已有实验；保留第二周完成标记。
- 主循环不得调用现成 `generate()`；参考库生成只能作为另行标注的辅助检查。
- GPU 测试前重新检查卡的负载，记录实际卡号/UUID。一次只加载必要模型，不并行启动多个大权重实例。
- 本阶段不编辑外部 nano-vLLM 两份源码，不把其已有能力列为自己的新增功能。
- 以下均不是第一阶段交付：多请求调度器、分页分配器、手写 DeltaNet kernel、CUDA Graph、分块 Prefill、抢占恢复、多模态输入、FP8/INT8/INT4、MTP、KV 压缩、多卡/TP、C++/InferLab 重构。
- 不安装或升级依赖；遇到不兼容，保留具体错误，先提出最小变更再确认。
- 所有代码命令从项目根目录执行；新模块使用 `python -m week3.<module>`，不依赖恰巧存在的 PYTHONPATH。旧第一周测试单独设置 `PYTHONPATH=week1`。
- 只准备脚本不等于真实模型验证通过；SKIP、诊断、数值通过、学习验收分别记录。

## Review Focus

1. 未知模型或畸形嵌套配置：报出具体字段，不回退到全注意力计算。归 Task 1。
2. 错 revision、半下载或缺 tokenizer/权重：运行脚本只报缺失资产，不能悄悄联网、混用文件或随机初始化。归 Task 3、4。
3. 空输入、批量输入、输入中已有 EOS、异常后复用请求：边界明确，新请求必须从新状态开始。归 Task 2、5。
4. 线性状态 dtype 与配置不同、张量 view 共享存储：按实际张量统计，不能把逻辑字节和物理存储简单混为一谈。归 Task 6。
5. NaN/Inf、零范数参考、top-1 近似并列、参数变更后沿用旧容差：不能输出虚假的 PASS。归 Task 7。

---

## 执行组织、文件与前置检查

本计划只实现第一阶段。依赖顺序为 Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8；Task 5 的 CPU 测试可在权重等待期间执行，但真实 GPU 验收必须等待 Task 3、4。

推荐在当前任务由同一个助手按课执行：先完成 Task 1–2 并教学，再准备资产和生成循环，不一次性跳到最后。每个代码任务采用“测试失败 → 最小实现 → 测试通过 → 检查差异”的循环。文中的测试片段是必须落入测试文件的用例，不是已经运行的结果。

测试片段中的 `def test_*(self)` 放入各自文件的 `unittest.TestCase` 子类，文件显式 `import unittest`；辅助替身函数/类放在模块级。不要把带 self 的函数孤立放在模块级导致 unittest 没有发现测试。回归输出需要核对实际执行数量及 SKIP 数。

| 文件（相对项目根目录） | 操作与单一职责 |
| --- | --- |
| `week3/inspect_model.py`、`week3/test_inspect_model.py` | 修改：配置解释、模板输入和离线单元测试 |
| `week3/fixtures/qwen35_4b_config_minimal.json` | 新增：官方配置的有来源的缩减测试样例，不冒充模型资产 |
| `week3/test_tokenizer_integration.py` | 新增：显式开启的真实 tokenizer 测试 |
| `week3/model_assets.py`、`week3/test_model_assets.py` | 新增：官方资产准备、revision/完整性清单；运行入口不隐式下载 |
| `week3/reference_runtime.py`、`week3/test_reference_runtime.py` | 新增：本地参考模型加载及单次纯文本 forward |
| `week3/minimal_generate.py`、`week3/test_minimal_generate.py` | 新增：自写生成循环及可控测试替身 |
| `week3/cache_inspection.py`、`week3/test_cache_inspection.py` | 新增：只读混合状态快照和内存统计 |
| `week3/verify_generation.py`、`week3/test_verify_generation.py` | 新增：teacher forcing、误差指标、校准与验收 |
| `week3/HISTORICAL_QWEN3_06B.md` | 新增：保留旧模型记录与归属，不伪造迁移结果 |
| `ROADMAP.md`、`week3/PLAN.md`、`week3/LESSON_01.md`、`week3/ENVIRONMENT.md` | 修改：路线图、当前学习入口、环境事实 |
| `week3/LESSON_02.md`、`week3/LESSON_03.md`、`week3/GENERATION_REPORT.md` | 新增：生成循环、混合状态/对拍教学与实测报告 |
| `week3/results/` | 原始 JSON/日志/校准配置；已有 `.gitignore` 的 `results/` 会忽略它 |

开始实施时，使用 using-git-worktrees 技能检查隔离方式。本地桌面的 Git 仓库不是远程 infer-engine，不能在错误仓库创建工作树。当前远程存在未提交的 ROADMAP、Week 2/3 计划和未跟踪的 Week 3 文件；不能从 HEAD 建空白工作树后丢失这些学习成果。若隔离复制这些文件，逐个核对内容/指纹，禁止全量 stash、清理或覆盖源目录。

先记录下面命令输出，不在本计划编写阶段执行测试或改代码：

```bash
cd /home/ubuntu/infer-engine
git status --short
git diff --stat
git diff --cached --name-only
git log -1 --format='%H %s'
sha256sum ROADMAP.md week2/PLAN.md week3/PLAN.md week3/inspect_model.py week3/test_inspect_model.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model -v
```

已有记录为 20 项 CPU 测试通过，实施时以新输出为准。缺本地旧 tokenizer 时先区分资产缺失和算法回归，不能删除失败用例来“变绿”。每个任务的提交只包含审阅过的本任务差异；禁止 `git add .`、自动 push 或把用户已有改动一并提交。提交前执行 `git diff --cached --check` 并查看暂存文件名单。

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

### Task 3：固定 revision 的模型资产准备与完整性检查

**Files:** Create `week3/model_assets.py`、`week3/test_model_assets.py`；Update `week3/ENVIRONMENT.md`。

**Interfaces:**

- Produces：`validate_assets(model_dir: Path, require_weights: bool) -> dict`，仅读本地，返回 manifest；缺失/混合 revision 抛 `ValueError`。
- Produces：`validated_shards(model_dir: Path) -> list[str]`，读取 `model.safetensors.index.json`，验证每个 shard 相对路径在目录内且后缀为 `.safetensors`，返回去重排序名单；空 weight_map 或非法路径抛 `ValueError`。
- Produces：`prepare_assets(model_dir: Path, revision: str, include_weights: bool, *, fetch=None) -> dict`；`fetch=None` 时延迟导入 `snapshot_download`，测试注入替身。revision 必须为 40 位十六进制 commit SHA。CLI `--revision SHA --metadata-only` 或 `--revision SHA --with-weights`，模式必须显式给出。
- manifest 文件名 `infer_engine_manifest.json`：repo_id、revision、mode、文件相对路径/size/SHA256、完整性状态、下载时间。单纯手写 manifest 不能代替验证实际文件。
- manifest 的 hash 清单仅覆盖下载白名单内的资产，不包含 manifest 自身、`.cache/`、临时文件和日志，避免递归 hash 或把下载器内部缓存当作模型内容。

- [ ] **Step 1：先写失败测试。** mock 下载器，只用临时目录；测试缺配置、缺 tokenizer、缺 shard、已有不同 revision、路径穿越和下载中断。

```python
import tempfile
from pathlib import Path
from unittest.mock import Mock
from week3.model_assets import prepare_assets, validate_assets

def test_invalid_revision_never_downloads(self):
    fetch = Mock()
    with tempfile.TemporaryDirectory() as folder:
        with self.assertRaisesRegex(ValueError, 'revision'):
            prepare_assets(Path(folder), 'main', False, fetch=fetch)
    fetch.assert_not_called()

def test_missing_local_assets(self):
    with tempfile.TemporaryDirectory() as folder:
        with self.assertRaises(ValueError):
            validate_assets(Path(folder), require_weights=True)
```

构造 index 的 `weight_map={'x':'../outside.safetensors'}` 时必须拒绝；构造两个 shard 只放一个，完整性不得为 complete。已有非空目录且没有本项目 manifest 时停止，不覆盖未知资产。下载失败不创建 complete 清单，不删除部分文件，允许仅在同 revision 下恢复。
- [ ] **Step 2：运行失败测试。** `CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_model_assets -v`。
- [ ] **Step 3：实现准备/校验。** 下载只允许官方仓库；metadata 选择 config/tokenizer/chat-template/generation-config/index，不下载 Python 远程代码。清单 SHA256 用分块读取计算，避免把大权重一次读入内存。权重文件以 index 的不同 shard 文件名为准，解析并验证所有解析路径位于 model_dir 内；只校验存在/大小不够，加载前复验清单 hash。

```python
import hashlib
import re
if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
    raise ValueError('revision 必须是固定 commit SHA')
if fetch is None:
    from huggingface_hub import snapshot_download
    fetch = snapshot_download
patterns = ['*.json', '*.jinja', '*.model', 'merges.txt', 'vocab.txt']
if include_weights:
    # 权重模式要求此前 metadata 已下载、同 revision 且验证通过。
    patterns += validated_shards(model_dir)
fetch(repo_id='Qwen/Qwen3.5-4B', revision=revision,
      local_dir=str(model_dir), allow_patterns=patterns)
```

磁盘检查使用 `shutil.disk_usage(model_dir.parent).free`，首次完整下载要求大于 index 记录的权重总字节加 2 GiB 余量；不能把磁盘大小当 GPU 需求。只有全部验证成功后用临时 manifest + replace 更新为 complete。JSON 写入拒绝 NaN；不打印凭据或完整环境变量。
- [ ] **Step 4：先单元测试，再执行 metadata 准备。** 下列网络命令属于实施阶段，当前计划编写不执行。若需要权限，使用权限机制；失败不得偷偷换非官方镜像。

```bash
QWEN35_REVISION=$(/home/ubuntu/enter/envs/nanovllm/bin/python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3.5-4B').sha)")
/home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.model_assets --model /home/ubuntu/huggingface/Qwen3.5-4B --revision "$QWEN35_REVISION" --metadata-only
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --model /home/ubuntu/huggingface/Qwen3.5-4B
RUN_TOKENIZER_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_tokenizer_integration -v
```

若目录已有同 revision 的本项目 manifest，直接使用其中 revision，不重新解析 main。metadata 未通过前不下载大权重；通过后使用同一个 `$QWEN35_REVISION` 执行 `--with-weights`，完整下载可耗时较长，单独汇报进度，不阻塞教学。
- [ ] **Step 5：更新环境事实与提交。** 只提交代码和环境记录，不提交权重或凭据。建议消息 `feat(week3): pin and validate Qwen3.5 model assets`。

### Task 4：参考模型加载与无隐藏位置依赖的 forward

**Files:** Create `week3/reference_runtime.py`、`week3/test_reference_runtime.py`；Update `week3/ENVIRONMENT.md`。

**Interfaces:**

- Consumes：`validate_assets(model_dir, require_weights=True)`。
- Produces：`load_reference(model_dir: Path, device: str, dtype_name: str = 'bf16') -> tuple`，返回 `(model, tokenizer, manifest, eos_ids)`；dtype_name 只接受 bf16/fp32，FP32 仅作后续诊断。
- Produces：`forward_last(model, ids, *, processed_tokens: int = 0, cache=None, use_cache: bool = True) -> tuple`，返回 `(last_logits, new_cache)`；last_logits 固定 `[1,V]`，cache 仅由调用者持有。
- CLI `python -m week3.reference_runtime --model PATH --device cuda:0 --smoke`：只做固定短文本的一次 Prefill 和一次 Decode，记录结果，不声称生成验收已完成。

- [ ] **Step 1：先写 CPU forward 替身测试。** 必须验证位置与 mask，不只验证输出形状。

```python
from types import SimpleNamespace
import torch
from week3.reference_runtime import forward_last

class CaptureModel:
    def __init__(self):
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=8))
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(logits=torch.zeros(1, 1, 8),
                               past_key_values=object() if kwargs['use_cache'] else None)

def test_decode_positions_and_mask(self):
    model = CaptureModel()
    marker = object()
    logits, state = forward_last(model, torch.tensor([[5]]),
                                processed_tokens=7, cache=marker)
    self.assertEqual(tuple(logits.shape), (1, 8))
    self.assertEqual(model.kwargs['position_ids'].tolist(), [[[7]], [[7]], [[7]], [[7]]])
    self.assertEqual(tuple(model.kwargs['attention_mask'].shape), (1, 8))
    self.assertIs(model.kwargs['past_key_values'], marker)
    self.assertNotIn('pixel_values', model.kwargs)
```

增加空输入、B=2、非 long 类型、token 越界、负 processed_tokens、`use_cache=False` 携带 cache、Decode 丢缓存、返回 NaN 或返回形状错误的拒绝测试。mock `from_pretrained` 返回语言权重 missing/mismatched 时加载必须失败；不能用安装依赖让测试通过。
- [ ] **Step 2：运行失败测试。** `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_reference_runtime -v`。
- [ ] **Step 3：实现加载与文本 forward。** 先验证资产，再加载完整官方 ConditionalGeneration checkpoint；优先 CPU 加载后显式 `.to(device)`，避免依赖自动多卡映射，先检查主机内存。`attn_implementation='eager'` 固定可追踪的完整注意力路径，不请求安装 FlashAttention/FLA。

```python
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_name]
model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False,
    dtype=dtype, attn_implementation='eager', output_loading_info=True)
# 检查 loading 后才允许移动到 GPU；关键权重 missing/mismatched/error 一律失败。
model = model.eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(str(model_dir),
    local_files_only=True, trust_remote_code=False)
```

校验 `loading` 中 missing_keys、mismatched_keys、error_msgs；意外权重默认也失败。若出现官方 checkpoint 的 `mtp.*` 多余权重，只有逐项核对其确属不使用的 MTP 后才允许这个确切前缀，并记录名单；不能用 `strict=False` 或宽泛忽略来绕过。EOS 优先读取 model.generation_config.eos_token_id，接受合法 int/list[int] 并统一为 set；缺失时读取 tokenizer.eos_token_id，再缺失即失败，排除 bool/负值/越界。

```python
# forward_last 的校验完成后：纯文本、无 padding 的四轴位置相同。
n = ids.shape[1]
positions = torch.arange(processed_tokens, processed_tokens + n, device=ids.device)
positions = positions.view(1, 1, n).expand(4, 1, n)
mask = torch.ones((1, processed_tokens + n), dtype=torch.long, device=ids.device)
with torch.inference_mode():
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                past_key_values=cache, use_cache=use_cache,
                logits_to_keep=1, return_dict=True)
    last_logits = out.logits[:, -1, :]
```

四轴位置来源是当前已检查的 Transformers 5.8.0 text forward；这样避免依赖包装层可变 rope_deltas 的隐式推断。版本变更后必须重新检查接口，不把本方案推广到图片输入。`use_cache=True` 必须实际返回 cache；完整重算必须 `cache=None, processed_tokens=0, use_cache=False`。输入中的真实 image/video 占位 token ID 报错，不启动视觉模块。
- [ ] **Step 4：CPU 通过后才做 GPU smoke。** 每次先查询资源，记录选中卡 UUID 和当前占用，不把这里的检查结果永久当作空闲保证。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

在服务器交互 shell 用 `read -r -p '本次确认可用的物理 GPU 编号: ' QWEN35_GPU` 设置本次变量，再执行下面命令；这是资源选择，不硬编码历史空闲卡。

```bash
CUDA_VISIBLE_DEVICES="$QWEN35_GPU" HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.reference_runtime --model /home/ubuntu/huggingface/Qwen3.5-4B --device cuda:0 --smoke
```

有其他任务或加载不兼容时不运行，保留错误。完整模型可能加载未使用的视觉权重，日志必须注明；首次两次 forward 仅检查 finite、shape、cache 存在和位置增长，不能报为完整正确性验收。
- [ ] **Step 5：提交并记录加载边界。** 建议消息 `feat(week3): add explicit text reference forward`。报告记录模型 revision、代码指纹、dtype、eager/DeltaNet 实际路径和加载告警。

### Task 5：自己的 Prefill/Decode 循环与生命周期

**Files:** Create `week3/minimal_generate.py`、`week3/test_minimal_generate.py`、`week3/LESSON_02.md`。

**Interfaces:**

- Consumes：Task 2 的 `inspect_prompt`；Task 4 的 `load_reference`、`forward_last`。
- Produces：`run_greedy(model, input_ids, eos_ids: set[int], max_new_tokens: int = 16, *, step=forward_last, observe_cache=None) -> dict`。observe_cache 若提供，签名为 `(cache, processed_tokens: int) -> dict`，只返回 JSON 元数据。
- 返回字段：`input_ids`、`output_ids`、`processed_tokens`、`generated_tokens`、`stop_reason`、`trace`；不返回缓存对象/GPU 张量。
- CLI：`--model`、`--prompt`、`--max-new-tokens`、`--device`、`--output`。输出文件已存在时拒绝覆盖；不提供隐式批量或 padding。

- [ ] **Step 1：先写控制流测试。** 测试替身使用与正式 forward 相同的签名，不加载权重。

```python
import torch
from week3.minimal_generate import run_greedy

def test_first_output_is_not_yet_cached(self):
    calls = []
    cache_marker = object()
    def fake_step(model, ids, *, processed_tokens=0, cache=None, use_cache=True):
        calls.append((ids.tolist(), processed_tokens, cache))
        scores = torch.full((1, 8), -10.0)
        scores[0, 3 if len(calls) == 1 else 7] = 10.0
        return scores, cache_marker
    r = run_greedy(None, torch.tensor([[7, 2]]), {7}, 4, step=fake_step)
    self.assertEqual(r['output_ids'], [3, 7])
    self.assertEqual(r['stop_reason'], 'eos')
    self.assertEqual(r['processed_tokens'], 3)
    self.assertEqual(r['generated_tokens'], 2)
    self.assertIsNone(calls[0][2])
    self.assertEqual(calls[1][:2], ([[3]], 2))
```

这个例子输入中已有 EOS=7，但必须生成到新采样出的 7 才停。增加 max_new_tokens=0 不调用 forward、=1 只一次 Prefill、达到长度上限、EOS list 规范化、max_new_tokens 为负/bool、B=2、空输入和非有限 logits 的测试。模拟第二步抛异常后再次调用，断言新请求第一次 cache=None、位置 0；不能在函数默认参数中放可变缓存。
- [ ] **Step 2：运行失败测试。** `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_minimal_generate -v`。
- [ ] **Step 3：实现最小循环。** 下列是控制核心；增加已明确的输入/shape/finite 校验、JSON trace 和 finally 清理，不引入 scheduler。

```python
cache = None
logits = None
current = input_ids
processed = 0
generated = []
trace = []
reason = 'length'
try:
    for index in range(max_new_tokens):
        logits, cache = step(model, current, processed_tokens=processed,
                             cache=cache, use_cache=True)
        processed += current.shape[1]
        token_id = int(logits.argmax(dim=-1).item())
        generated.append(token_id)
        event = dict(phase='prefill' if index == 0 else 'decode',
                     input_ids=current.tolist()[0], processed_tokens=processed,
                     token_id=token_id, generated_tokens=len(generated))
        if observe_cache is not None:
            event['cache'] = observe_cache(cache, processed)
        trace.append(event)
        if token_id in eos_ids:
            reason = 'eos'
            break
        current = input_ids.new_tensor([[token_id]])
finally:
    cache = None
    logits = None
```

return 构造仅使用 Python 值，trace 增加本次 position 起止值与停止原因。异常原样携带阶段信息向上传播，CLI 非零退出；不把异常转成正常的“length”。日志不持有 cache，release 不等于清空 CUDA allocator。缓存仅在成功 forward 后推进计数。
- [ ] **Step 4：CPU 通过后运行三条短提示词。** 固定为“用一句话解释 KV Cache。”“计算 2+3，只输出结果。”“What is prefill in language model inference?”；每条 max_new_tokens=16、同模板、贪心。EOS 自然未触发时保留长度停止，不能改输出假装成功；EOS 分支由替身测试证明。

```bash
CUDA_VISIBLE_DEVICES="$QWEN35_GPU" HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.minimal_generate --model /home/ubuntu/huggingface/Qwen3.5-4B --device cuda:0 --prompt '用一句话解释 KV Cache。' --max-new-tokens 16 --output week3/results/qwen35-generate-kv.json
```

另两条使用各自文件名 `qwen35-generate-math.json`、`qwen35-generate-prefill.json`。不以文本是否优美作为数值正确性证据；16 token 截断不等于模型故障。
- [ ] **Step 5：讲义、差异审阅与提交。** LESSON_02 用真实 trace 解释首 token 来自 Prefill 和 `processed=P+N-1` 的成立条件。建议消息 `feat(week3): implement a manual cached generation loop`。请用户手工推演一次 EOS 和一次长度停止。

### Task 6：混合状态只读快照与实际显存分类

**Files:** Create `week3/cache_inspection.py`、`week3/test_cache_inspection.py`；Modify `week3/minimal_generate.py`；Create `week3/LESSON_03.md` 的状态部分。

**Interfaces:**

- Produces：`snapshot_cache(cache, layer_types: list[str], processed_tokens: int) -> dict`，返回 `layers`、`logical_tensor_bytes`、`unique_storage_bytes`、`processed_tokens`。
- 每层记录原始 `layer_index`、`layer_type` 和 tensors（名称、shape、dtype、device、logical_bytes）；未初始化张量为 null，不假装 0 字节已验证。
- Consumes：Transformers 5.8.0 `cache.layers[i]`：Full Attention 为 `keys/values`，linear 为 `conv_states/recurrent_states`。类型不符/接口缺失必须明确失败，不能吞错跳层。

- [ ] **Step 1：先写共享存储/混合 dtype 测试。** 只用 CPU 小张量，不分配正式缓存。

```python
from types import SimpleNamespace
import torch
from week3.cache_inspection import snapshot_cache

def test_observe_actual_dtype_and_shared_storage(self):
    base = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
    conv = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    recurrent = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    cache = SimpleNamespace(layers=[
        SimpleNamespace(keys=base, values=base.view_as(base)),
        SimpleNamespace(conv_states=conv, recurrent_states=recurrent)])
    r = snapshot_cache(cache, ['full_attention', 'linear_attention'], 2)
    self.assertEqual(r['logical_tensor_bytes'], 48)
    self.assertEqual(r['unique_storage_bytes'], 40)
    self.assertEqual(r['layers'][1]['tensors']['recurrent_states']['dtype'], 'torch.float32')
    self.assertTrue(torch.equal(base, torch.zeros_like(base)))
```

再把 recurrent 改 BF16，输出必须随实际张量改变，而不是硬编码配置中的 FP32。增加 cache=None、层数不匹配、缺键、未初始化 None 和只读前后数值相等测试。
- [ ] **Step 2：运行失败测试。** `CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_cache_inspection -v`。
- [ ] **Step 3：实现只读快照。** 不把 CUDA tensor 存进结果；logical 指张量元素字节，storage 指不同底层存储的总大小，不是 allocator 的 reserved 字节。

```python
logical_bytes = tensor.numel() * tensor.element_size()
storage = tensor.untyped_storage()
storage_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
if storage_key not in seen_storages:
    seen_storages.add(storage_key)
    unique_storage_bytes += storage.nbytes()
```

每个调用的 seen_storages 独立；不能跨请求累计旧地址。缺失状态用 null 并将整体快照完整性标为 incomplete；不要把未初始化与已分配 0 元素张量混同。Full Attention 长度从实际 keys.shape[-2] 读取，并核对等于 processed_tokens；不要对线性层调用不支持的 get_seq_length(i)。
- [ ] **Step 4：接入观察回调并运行短实验。** `observe_cache=lambda cache, n: snapshot_cache(cache, layer_types, n)`，layer_types 从实际模型 text_config 取得。对 Prefill 和 3 次 Decode 记录 8 层 K/V 增长与 24 层线性状态形状；另用 A=较长提示词、B=短提示词验证 B 从新状态开始。

显存分别记录：权重参数实际字节、逻辑状态字节、唯一存储字节、`torch.cuda.memory_allocated()`、`memory_reserved()` 和 peak allocated。加载峰值和请求峰值分开：加载完成同步后重置请求 peak，再开始请求。带日志数据不当作性能基线，清理引用后 reserved 仍高不是泄漏证据。
- [ ] **Step 5：讲义说明、回归与提交。** 建议消息 `feat(week3): inspect hybrid cache states without mutation`。让用户解释“状态形状固定，为什么每步仍要更新内容”，但不要求本阶段手推 Delta Rule 全部数学。

### Task 7：固定 token 的数值对拍、容差冻结与状态隔离

**Files:** Create `week3/verify_generation.py`、`week3/test_verify_generation.py`；Update `week3/LESSON_03.md`；运行产物写入 `week3/results/`，不提交大张量。

**Interfaces:**

- Consumes：`forward_last`、`snapshot_cache`、模型 manifest。
- Produces：`compare_logits(reference, candidate) -> dict`，输出 max_abs、mean_abs、relative_l2（零分母且非零误差时为 null）、zero_reference、top1_reference、top1_candidate、reference_margin、candidate_margin。
- Produces：`evaluate_metrics(metrics: dict, limits: dict | None) -> str`，只返回 `diagnostic`、`pass`、`fail`、`needs_review`。
- Produces：`validate_tolerances(document: dict, expected_fingerprint: dict) -> dict`，检查 approved、三个有限非负阈值、非空 review_reason 和所有指纹一致，返回阈值 dict；不满足时抛 `ValueError`。
- Produces：`verify_prefixes(model, prompt_ids, continuation_ids, *, limits=None, step=forward_last) -> dict`，每个固定前缀一条记录；continuation_ids 为 `[1,N]`，N>=1。
- Produces：`verify_isolation(model, prompt_a, prompt_b, *, limits=None, step=forward_last) -> dict`，比较 B 单独运行、A 结束后 B、A 异常后 B。
- CLI 模式互斥：`--calibrate` 或 `--verify`；`--verify` 无 `--tolerance-file` 时只做诊断，不产生通过结论。全部结果带输入/模型/代码/环境指纹。

- [ ] **Step 1：先写指标与判定失败测试。** 不依赖真实模型，数值误差在 FP32 中计算。

```python
import torch
from week3.verify_generation import compare_logits, evaluate_metrics, verify_prefixes, validate_tolerances

def test_zero_norm_and_nonfinite(self):
    zero = torch.zeros(1, 3)
    m = compare_logits(zero, zero)
    self.assertEqual((m['max_abs'], m['relative_l2']), (0.0, 0.0))
    m = compare_logits(zero, torch.ones(1, 3))
    self.assertIsNone(m['relative_l2'])
    self.assertTrue(m['zero_reference'])
    for value in (float('nan'), float('inf')):
        with self.assertRaises(ValueError):
            compare_logits(zero, torch.tensor([[0.0, 0.0, value]]))

def test_close_logits_do_not_hide_top1_change(self):
    m = compare_logits(torch.tensor([[1.0, 0.999]]), torch.tensor([[0.999, 1.0]]))
    limits = dict(max_abs=0.01, mean_abs=0.01, relative_l2=0.01)
    self.assertEqual(evaluate_metrics(m, None), 'diagnostic')
    self.assertEqual(evaluate_metrics(m, limits), 'needs_review')

def test_unapproved_or_wrong_revision_limits(self):
    fingerprint = {'revision': 'a' * 40, 'dtype': 'bf16'}
    doc = {'approved': False, 'review_reason': '独立参考诊断',
           'limits': {'max_abs': 0.01, 'mean_abs': 0.01, 'relative_l2': 0.01},
           'fingerprint': fingerprint.copy()}
    with self.assertRaises(ValueError):
        validate_tolerances(doc, fingerprint)
    doc['approved'] = True
    doc['fingerprint']['revision'] = 'b' * 40
    with self.assertRaises(ValueError):
        validate_tolerances(doc, fingerprint)
```

测试 shape 不同、空 logits、非法/非有限容差、误差超阈值、reference 全零而 candidate 非零必须 fail。额外测试容差文件 `approved=false`、revision/dtype/backend/代码指纹不一致时拒绝用作验收。近似并列只能解释为何不一致，不能自动忽略后写 PASS。
- [ ] **Step 2：写 teacher-forcing 的状态替身测试并运行 red。** 替身的缓存保存历史 token 求和，正确版本应全重算/增量一致，故意污染缓存版本必须被检测。

```python
def sum_step(model, ids, *, processed_tokens=0, cache=None, use_cache=True):
    total = int(ids.sum().item()) + (0 if cache is None else cache['sum'])
    scores = torch.zeros(1, 8)
    scores[0, total % 8] = 2.0
    return scores, ({'sum': total} if use_cache else None)

def test_fixed_prefix_verification(self):
    r = verify_prefixes(None, torch.tensor([[1, 2]]), torch.tensor([[3, 4]]),
                        limits=dict(max_abs=0.0, mean_abs=0.0, relative_l2=0.0),
                        step=sum_step)
    self.assertEqual(len(r['steps']), 3)
    self.assertTrue(all(s['status'] == 'pass' for s in r['steps']))
```

污染版本为 sum_step 的包装：cache 非 None 时先把其 sum 加 1，再调用 sum_step。应至少一个 step 非 pass，且 trace 证明两边始终使用相同前缀，不是独立生成后比较文本。

Run：`CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_verify_generation -v`。预期在缺失函数/新行为处失败。

- [ ] **Step 3：实现指标与对拍。** 核心流程如下，添加每一步输入/位置/缓存摘要和异常清理，禁止两条路径共享同一可变缓存。

```python
# compare_logits：先检查 shape、维度、finite，再将两边转为 float32。
diff = candidate.float() - reference.float()
max_abs = float(diff.abs().max().item())
mean_abs = float(diff.abs().mean().item())
norm = float(torch.linalg.vector_norm(reference.float()).item())
diff_norm = float(torch.linalg.vector_norm(diff).item())
relative_l2 = diff_norm / norm if norm != 0 else (0.0 if diff_norm == 0 else None)
top2 = torch.topk(reference.float(), k=2, dim=-1).values
reference_margin = float((top2[0, 0] - top2[0, 1]).item())
```

V<2 时明确拒绝，不让 topk 抛难以理解的底层错误。先进行完整 Prefill 的 `use_cache=False` 和 `True` 比较；随后执行：

```python
prefix = prompt_ids
processed = prompt_ids.shape[1]
# cache 来自已比较过的缓存 Prefill；每轮 reference 都从空状态重算。
for j in range(continuation_ids.shape[1]):
    token = continuation_ids[:, j:j + 1]
    prefix = torch.cat([prefix, token], dim=1)
    reference, _ = step(model, prefix, processed_tokens=0, cache=None, use_cache=False)
    candidate, cache = step(model, token, processed_tokens=processed, cache=cache, use_cache=True)
    processed += 1
    metrics = compare_logits(reference, candidate)
```

任何 NaN/Inf 或输入错误令整次验证失败。所有 step pass 才允许总状态 pass；只要出现 diagnostic/needs_review，总状态不能为 pass。数值违反阈值返回非零退出码，缺资产返回准备错误；合法诊断模式退出成功但 JSON `passed=false,status='diagnostic'`。

- [ ] **Step 4：独立校准，生成未批准的候选容差。** 不用验收提示词的缓存路径误差来自动放宽门槛。校准文本固定为“简要解释矩阵乘法。”和“Name two primary colors.”；取各自完整前缀以及附加 tokenizer 编码“缓存复用历史信息。”的前 4 个 token 的前缀。编码不足 4 个 token 则报准备错误，不补随机 token。

分别用 BF16 完整重算两次检查重复稳定性，再单独加载 FP32 完整重算比较。两个精度的模型顺序加载并显式释放，不同时占用两份 GPU 权重；只保存短前缀最后位置 logits 的 CPU 数据。更换 dtype 后记录完整环境和真实状态 dtype。FP32 无法运行时保持 diagnostic，不擅自省略这一校准依据。

候选上限由独立诊断数据产生，计算方法固定如下（系数是工程筛查的保守余量，不是数学等价证明）：

```python
candidate_limits = {
    'max_abs': max(1e-4, 4 * max(row['max_abs'] for row in calibration_rows)),
    'mean_abs': max(1e-5, 4 * max(row['mean_abs'] for row in calibration_rows)),
    'relative_l2': max(1e-5, 4 * max(row['relative_l2'] for row in calibration_rows)),
}
```

relative_l2 无法定义或参考含非有限值时，不生成候选容差。输出 `approved=false` 的 tolerance JSON，包含候选值、每项最大值来自哪一前缀、两次 BF16 的误差、BF16/FP32 的误差、模型 SHA、config/tokenizer hash、dtype、torch/transformers、attention backend、reference/verification 脚本 hash、校准数据 hash、理由。

在运行验收提示词之前审阅并冻结该文件：若参考路径不稳定或候选门槛相对 top-1 margin 过宽，先调查，不批准。批准仅改变 approved 与 review_reason，不自动修改候选值。此处是数值标准审阅，不是预先承诺任何浮点阈值必然通过；阈值更改生成新版本，旧失败记录保留。

- [ ] **Step 5：用冻结容差运行真实对拍与隔离。** 验收使用 Task 5 的三条提示词，各比较 Prefill + 4 次固定 Decode，额外覆盖一个有效普通 token 的 T=1 原始输入（不从特殊占位 token 中挑选）。每条保存实际 IDs，不按中文字符数推测长度。

```bash
CUDA_VISIBLE_DEVICES="$QWEN35_GPU" HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.verify_generation --model /home/ubuntu/huggingface/Qwen3.5-4B --device cuda:0 --calibrate --output week3/results/qwen35-calibration.json
CUDA_VISIBLE_DEVICES="$QWEN35_GPU" HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.verify_generation --model /home/ubuntu/huggingface/Qwen3.5-4B --device cuda:0 --verify --tolerance-file week3/results/qwen35-tolerances.json --output week3/results/qwen35-verification.json
```

校准命令另外创建同目录的 `qwen35-tolerances.json`，初始为未批准，已存在即拒绝覆盖。第二条只能在审阅冻结后执行。跨进程不会自动保留 `$QWEN35_GPU`，执行前重新选择与核查。

隔离验证复用同一个参考模型实例，先运行 B 并保存最后 logits，再运行 A 后重新运行 B；再用 step 包装让 A 完成一次有状态 forward 后抛出指定 RuntimeError，在捕获并清理后重新运行 B。三次 B 的 Prefill/固定 Decode 结果按冻结标准比较，首步必须无缓存、位置 0。故障注入仅用于验证，不修改第三方模型源文件。

- [ ] **Step 6：讲义、回归、提交。** 建议消息 `test(week3): verify Qwen3.5 cached logits and request isolation`。请用户解释 teacher forcing、近似并列 top-1 和为什么不能用两段相似文字代替数值对拍；未通过的用例保留原始数据并阻止本阶段完成标记。

### Task 8：回归、事实报告与学习验收

**Files:** Create `week3/GENERATION_REPORT.md`；Update `week3/PLAN.md`、`week3/ENVIRONMENT.md`、`ROADMAP.md`、`week3/LESSON_02.md`、`week3/LESSON_03.md`。

**Interfaces:** Consumes 前七个任务的 JSON 结果、固定版本和日志；Produces 可复现报告与逐项验收状态，不新增运行时接口。

- [ ] **Step 1：运行完整无 GPU 回归。** 这是正式命令，不以局部测试或旧的“20 项”记录代替新输出。集成测试的 SKIP 要单列。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model week3.test_model_assets week3.test_reference_runtime week3.test_minimal_generate week3.test_cache_inspection week3.test_verify_generation -v
RUN_TOKENIZER_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_tokenizer_integration -v
git diff --check
```

- [ ] **Step 2：检查原始数据完整性。** 必须有三条生成记录、校准记录、已冻结容差、逐步验证及隔离记录；JSON 禁止 NaN/Infinity。核对模型/环境/代码指纹一致，所有 `passed` 从明细汇总，不能手填 true。若缺少任何记录，报告说明对应项未完成。
- [ ] **Step 3：撰写有事实依据的报告。** 必须包含下列标题及对应内容，数字只能从实际输出填写：

```markdown
# Qwen3.5-4B 单请求参考生成与混合缓存正确性报告

## 完成范围与复用边界

## 环境、模型 revision 与复现命令

## 输入模板、token IDs 与停止原因

## Prefill/Decode 逐步 logits 误差与冻结标准

## Full Attention K/V、线性状态与实际显存

## 正常结束及异常后的请求隔离

## 失败记录、限制与尚未完成项

## 下一阶段进入条件
```

结果表每行至少有 prompt/case、输入 token 数、处理 token 数、生成 token 数、停止原因、数值状态、数据文件与 SHA256。权重加载峰值、请求峰值和理论 128 MiB 分开；此轮无吞吐/TTFT 优化结论，不能从带日志的整段时间算 TPOT。`results/` 被忽略但必须实际保存在服务器；报告写出其路径、hash 和重新生成命令，不称为“clone 后自带原始数据”。
- [ ] **Step 4：进行学习验收。** 让用户独立回答：两种状态分别保存什么；为什么只给 8 层算普通 KV；第一个输出 token 何时进入缓存；为何需要固定相同下一 token 对拍；A 结束后 B 应从哪些状态开始。只完成脚本而用户尚未讲清时，工程验证与学习验收分别打勾。
- [ ] **Step 5：审查并提交本阶段结果。** 使用 verification-before-completion 对照实际测试输出。建议消息 `docs(week3): report the Qwen3.5 reference baseline`。任何 GPU/数值/隔离项缺失，保留 W3 进行中，不宣称已完成 nano-vLLM 原生适配。

## 计划自检与交接

| 设计要求 | 对应任务 |
| --- | --- |
| 模型类型、嵌套配置、8/24 层、理论 KV | 1 |
| 单次模板、真实 IDs、形状教学、历史成果保留 | 2 |
| 固定 revision、资产完整性、不隐式联网 | 3 |
| 本地完整 checkpoint、位置/mask、加载告警与 GPU 边界 | 4 |
| 自写循环、EOS/长度停止、异常清理 | 5 |
| 混合状态 shape/dtype、共享存储、显存分类 | 6 |
| teacher forcing、容差依据、状态隔离、失败不误报 | 7 |
| 三条真实提示词、回归、事实报告、学习验收 | 5、7、8 |

阶段完成不意味着下一阶段已获实现批准：原生 nano-vLLM 接入和多请求混合状态管理仍需依据本阶段结果单独设计。性能目标由测量决定，不继承简历截图的功能或百分比。

执行前请用户审阅本计划并选择方式：

- **当前任务内逐课执行（推荐）**：同一个助手按任务执行与教学，阶段末检查整体差异；接口依赖紧密，便于用户理解。
- **子代理实现并独立审查**：每个任务分派实现与审查，最后整体检查；上下文与审查成本更高，教学仍由主助手串联。

本轮仅编写计划。没有生成生产模块、下载权重、变更依赖、运行 GPU 验证或提前把验收勾选为完成。
