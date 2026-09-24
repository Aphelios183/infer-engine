# infer-engine 八周路线图

## 最终交付与判断标准

交付物是 **infer-engine** —— 一个能跑真实模型、有真实性能数字、每一行都能解释的
小型推理引擎。

判断标准不是"我学完了 AI Infra"，而是能拿着这个仓库说清：

> 我理解并跑通了推理链路，定位了一个实际瓶颈，完成了改进，验证了正确性，
> 并知道这个改进在哪些条件下有效。

## 固定学习节奏

```
先预测结果 → 自己写或修改 → 运行验证 → 解释差异
```

每一步都要留下可复现的证据（数字、日志、提交）。

## 八周计划

| 周 | 要回答的问题 | 任务 | 验收标准 |
| --- | --- | --- | --- |
| 1 | Attention 到底算了什么 | 读懂 MHA、推导形状、补边界用例 | 不看代码讲清 Q/K/V、mask、softmax、多头拆分 |
| 2 | 为什么生成时不用重算全部历史 | 给教学 MHA 加 KV Cache，对比全量重算与增量计算 | 结果对齐；能算出 KV 显存占用 |
| 3 | 如何从 Attention 走到文字生成 | 跑通小模型，跟踪 tokenizer → 模型 → logits → 采样 | 能解释一个请求从输入到结束的全过程 |
| 4 | 多个长短不同的请求如何一起执行 | 跟踪调度器，加请求生命周期记录 | 能手工推演三个不同长度请求的调度过程 |
| 5 | KV 如何按块分配和共享 | 块管理实验：块表、引用计数、回收 | 能解释逻辑 token 到物理位置的映射，验证无提前释放或泄漏 |
| 6 | 时间实际花在哪里 | 建基准、逐请求计时、学 profiler、选一个瓶颈 | 有可重复的基线和支持判断的性能证据 |
| 7 | 怎样改善这个瓶颈 | 改一个策略或热点，做正确性与对照实验 | 能解释改动、收益、成本和退化场景 |
| 8 | 如何让别人相信和复现 | 整理代码、环境、命令、原始数据和报告 | 换个人能复跑；能独立修改并回答追问 |

第 1–2 周自己实现机制；第 3 周起以 nano-vLLM 为可读的真实运行时。

## 进入 C++ / InferLab 阶段的五个门槛

满足以下条件后再深入 C++ 与 InferLab（预计第 5 周）：

- 能独立解释 MHA 的 Q、K、V、causal mask 和各 tensor shape
- 手写 KV Cache，并验证缓存 decode 与完整重算结果一致
- 在 `nanovllm` 环境里跑通一个小模型生成
- 能画出一次请求的 prefill、decode、结束和 KV 释放过程
- 能读懂 nano-vLLM 的 `scheduler.py`、`block_manager.py`、`model_runner.py` 的主要逻辑

## 分工

- **nano-vLLM**：理解"真实推理引擎怎样工作"（第 1–4 周主力）
- **InferLab**：参考实现，用于对照与测量（第 3 周起旁读，第 5 周起深入）
- **交付物始终是 infer-engine**：InferLab 是参考答案和对照基线，不是要优化的对象

InferLab 中值得读的部分（第 5 周起，不要一开始就编译整个仓库）：

```
native_core/include/inferlab/paged_kv_cache.hpp
native_core/src/paged_kv_cache.cpp
native_core/include/inferlab/continuous_batch_scheduler.hpp
native_core/src/continuous_batch_scheduler.cpp
native_core/src/decoder_kv.cpp
```

## 第 6–7 周主攻方向（已选定）

> 长提示词请求进入后，正在生成的请求为什么会停顿？如何改善这种干扰？

限定为一项真实改动，例如调整 prefill 与 decode 的轮次分配。

**前置要求**：先确认所用 nano-vLLM 版本已有何种调度逻辑，不能把上游已有功能
当作自己的新增成果。

必须记录的指标：

- 首 token 延迟（TTFT）：提交请求后多久产生第一个 token
- 生成间隔（TPOT）：相邻输出 token 之间的等待时间
- 总吞吐：单位时间生成的输出 token 数
- 失败率与显存占用：是否通过牺牲容量或稳定性换取速度

负载类型：短请求、长请求、长短混合三类对照。

收益不必在所有场景都成立，但必须能解释为什么。

## 范围控制（明确不做）

- 八卡并行、张量并行
- 完整量化系统
- Android / 端侧运行时
- 多后端同时适配（TensorRT-LLM、KServe、端云路由）
- InferLab 的 evidence / governance / schema 层

第一版以单卡为主。C++/CUDA 随主线补基础。

## 第二周当前计划

2026-09-24：已完成三次 L40 Prefill/Decode 对照并讨论结果，见 [实验报告](week2/LESSON_03_RESULTS.md)。已学习容量、利用率与 GQA 估算；最终综合题尚有未完整作答项，随第三周模型配置练习复习，不标记为全项验收通过。

## 第三周当前计划

按用户要求进入 Week 3，详见 [Week 3：真实模型生成链路](week3/PLAN.md)。从已有 Qwen3-0.6B、本地 nano-vLLM 源码和 nanovllm 环境开始；先核查导入与兼容性，再做参考生成、缓存 logits 对拍和单请求跟踪。当前只完成计划与只读核查，尚未启动第三周模型推理。

## 进度


- [x] 环境就绪：`nanovllm` 环境（Python 3.11 + torch 2.5.1+cu124 + flash_attn 2.8.3）
- [x] W1 MHA 实现：逐步拆解脚本、可复用模块、数值对拍、因果性检查、T=1 边界用例
- [ ] W1 GPU 计时观察（seq_len 从 1 到 1024，理解 kernel launch 开销）
- [x] W2 KV Cache：全量重算 vs 增量计算对照实验（报告见 week2/LESSON_03_RESULTS.md）
- [ ] W3 真实模型生成：计划已写入 week3/PLAN.md，待执行和验收
