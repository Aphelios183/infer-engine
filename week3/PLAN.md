# Week 3：Qwen3.5 从配置到单请求生成

更新：2026-09-28。主线模型改为 **Qwen3.5-4B**，仍使用 nanovllm 环境。第二周已完成；旧 Qwen3-0.6B 实验仅保留为历史，不再作为当前课程验收。

当前课程工作区：`/home/ubuntu/infer-engine`，分支 `main`。最新课程已整合回主目录；直接从这里运行本课命令。

## 学习方式

每课按“读概念 → 预测结果 → 自己动手 → 运行验证 → 解释差异 → 验收”推进。
脚本测试通过不等于你已经掌握；老师提供检查工具，你完成关键推导和练习。
当前进入 [Lesson 2完整书写](LESSON_02.md)：真实forward与手写生成循环已接通。Lesson 1B 核心问答已完成，真实 tokenizer 已验证；层分类手写代码未提交，按用户选择不阻塞进入下一课。

## 课程与工程任务对应

| 课程 | 核心问题 | 你要动手做什么 | 工程任务与进入条件 |
| --- | --- | --- | --- |
| Lesson 1A | 混合模型为什么不能全层套 KV 公式？ | 推导形状、枚举层索引、计算 Full Attention KV | Tasks 1–2：配置解析与离线检查工具已验证；完成本课问答 |
| Lesson 1B | 文本怎样变成模型输入？ | 对比原文/模板、IDs、EOS、thinking 设置，检查编码一致性 | Task 2 工具 + Task 3 资产准备：真实 tokenizer 已验证；ModelScope 固定版本资产已下载，完整自动化资产校验模块仍待实现 |
| Lesson 2（当前） | 第一个输出 token 从哪里来？ | 先读一次 forward，再亲手写 prefill/decode/停止循环 | Tasks 3–5：固定模型资产，参考 forward，手写生成循环 |
| Lesson 3 | 混合模型到底缓存了什么？ | 检查 KV 与 recurrent/conv state 的形状、dtype、存储及请求隔离 | Task 6：实际状态快照，不只读配置猜测 |
| Lesson 4 | 增量计算真的正确吗？ | 固定相同 token 前缀，逐步比较全重算/缓存路径 logits | Task 7：teacher forcing，独立校准并冻结容差 |
| 本周验收 | 别人能否复现你的结论？ | 三条短提示词、停止原因、环境与原始结果报告 | Task 8：可复现实验与限制说明 |

设计与详细实现计划供查阅，不要求初学者先读完：
[设计](../archive/development/superpowers/specs/2026-09-25-qwen35-text-baseline-design.md)、
[实现计划](../archive/development/superpowers/plans/2026-09-25-qwen35-text-baseline.md)。

## 现在从哪里开始

当前先读 [Lesson 02](LESSON_02.md) 第9节，并阅读minimal_generate.py的run_greedy。真实运行见 [生成报告](LESSON_02_GENERATION_RESULTS.md)。以下命令保留为第一课复习：

```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --config-only --config-file week3/fixtures/qwen35_4b_config_minimal.json
```

这是裁剪后的教学配置，不是已下载的模型：输出中 `is_teaching_fixture=true`。
重点找出 `hidden_size`、两类层的数量与原始索引、`kv_mib`、`linear_state_bytes`。
这条配置复习命令不用GPU。真实权重已下载，但尚未加载，也未改变依赖。

你手写一段遍历 `layer_types` 的代码，将两类层的原始索引分别列出；先自己写，再与脚本输出对照。不要修改 fixture 来“配合答案”。

## 两种进度必须分开

工程进度：

- [x] 显式区分 Qwen3 与 Qwen3.5 嵌套配置，验证维度与层类型。
- [x] 只按 Full Attention 层计算 KV，保留原始层索引。
- [x] config-only 路径不导入 torch/transformers。
- [x] tokenizer 一致性检查工具及模拟 tokenizer 测试。
- [x] Qwen3.5-4B 的真实 tokenizer 资产与集成验证。
- [x] ModelScope 固定 revision 下载，Git LFS 完整性校验。
- [ ] 通用资产 manifest/校验模块（原 Task 3）仍待实现，下载操作不代表整个任务完成。
- [x] 权重加载、真实 forward、手写生成循环；三条短提示词验证完成。
- [ ] 全重算/缓存数值对拍与请求状态隔离验收（未完成）。
- [ ] 混合状态快照、逐步 logits 对拍、验收报告。

你的掌握情况（答题后逐项确认，不因为工具已写好而打勾）：

- [x] 能写出 token IDs、embedding 输出与单 token decode 的形状。
- [x] 能用实际 head_dim 推导 Full Attention KV 字节数。
- [x] 能解释未测的 linear state 不等于零内存。
- [ ] 能解释 chat template、tokenizer、EOS 和 thinking 开关。
- [x] 能解释 prefill 产生首 token，以及该 token 何时写入状态。
- [ ] 能解释为何数值对拍要让两条路径接收同一个 token。

## 验证命令与下一课门槛

本课离线单元测试：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_inspect_model -v
```

默认跳过的真实 tokenizer 测试不算通过；2026-09-28 已显式运行新模型集成测试并通过。下一步逐行读懂并修改自己的循环，再进入混合状态观察与数值对拍。不要复用旧模型 token 数和 EOS。

## 边界与后续衔接

本周先建立 Transformers 参考路径和自己的控制循环；这不等于已经完成 nano-vLLM 原生 Qwen3.5 适配。随后逐步把模型执行、混合状态管理、请求生命周期接入引擎。
先单卡、单请求、短文本，正确性通过后再增加并发；暂不做多模态、FP8、MTP、KV 压缩、多卡与 C++ 重写。

既有环境保持不变。加载模型前重新检查 GPU 使用情况，不沿用过去的空闲卡编号。
完整记录见 [环境与验证](ENVIRONMENT.md)；旧课保存在 [Qwen3 历史记录](../archive/old-models/QWEN3_06B.md)。
两个月是时间预算，不是全部高级功能都必须完成的承诺；若混合状态正确性未通过，先收缩范围，不跳过验收。
