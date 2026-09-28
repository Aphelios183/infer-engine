# Lesson 2：完整生成循环验证

日期：2026-09-28。代码：minimal_generate.py（自己的控制循环）与 reference_runtime.py（真实模型forward封装）。

## 验证范围

- 新增CPU单元测试21项通过；全课程回归发现61项，59项通过、2项资产集成测试默认跳过。真实模型运行另列，不把SKIP当通过。
- 首轮新增20项测试先失败后通过；审阅指出的未跟踪模型文件问题增加1项测试，先复现两个子用例失败，再修复并回归。
- 实际单卡生成三条提示词，最多16个新token，贪心、关闭thinking、batch=1、无padding。
- 没有调用model.generate()，没有修改依赖，没有吞吐或延迟优化结论。

## 真实输出

| 提示词 | 输入token | 新token | processed_tokens | 停止原因 | 可见输出 |
| --- | --- | --- | --- | --- | --- |
| 用一句话解释 KV Cache。 | 18 | 16 | 33 | length | KV Cache 是一种在生成式 AI 推理中通过缓存已计算好的 |
| 计算 2+3，只输出结果。 | 22 | 4 | 25 | eos | 5（后跟换行） |
| What is prefill in language model inference? | 21 | 16 | 36 | length | In the context of language model inference, **prefill** refers to the initial |

长度停止输出不完整是预算限制，不是伪造完整答案。
所有trace都满足：forward次数=新token数；processed_tokens=输入长度+新token数-1。
三条结果分别保存在 results/qwen35-generate-kv.json、qwen35-generate-math.json、qwen35-generate-prefill.json。
后两条复用同一个已加载模型，但每次run_greedy创建自己的cache；这不代替未来的独立运行数值隔离对拍。

## 运行条件

物理GPU5，UUID GPU-093b8773-f073-4c18-87c7-2afa616fd9c2；运行前利用率0%、已有289 MiB占用，非独占。
Qwen3.5-4B，ModelScope Git revision ed182e32090db791077e12e0f58d22f3daafa173。
完整ConditionalGeneration权重（含未使用的视觉部分），BF16、eval、inference_mode。
Full Attention eager；缺少线性注意力加速依赖，使用环境已有torch回退。
四轴纯文本position_ids显式相同；Decode mask覆盖历史长度和新输入。

## 不能混淆的两个结束标记

本次model.generation_config.eos_token_id实际为248044，优先用于生成停止。
tokenizer.eos_token_id为248046，是本次模板中的消息结束标记。
因此不要把之前模板实验的EOS常量直接复制进生成循环。模型配置没有EOS时才回退到tokenizer配置。

## 当前边界与审阅记录

- 未完成全重算/缓存逐步logits对拍；不声称数值正确性或原生nano-vLLM适配通过。
- 原计划通用资产下载器/manifest未实现；加载前采用固定Git/LFS仓库校验，拒绝额外文件与symlink。
- 独立审阅的重要发现已修复：未跟踪文件（包括额外单文件权重或generation_config）不能覆盖固定快照语义。
- 暂缓小项：更多加载失败路径的集成单测、同一测试替身的连续成功/失败请求隔离测试。
- 没有独立reference_runtime --smoke命令；请使用minimal_generate CLI的--max-new-tokens 2验证两次forward，不能把导入模块成功视为smoke通过。
- 本课仅交付最小生成循环，整个Week 3与原工程计划尚未完成。

## 阅读和复现

先读minimal_generate.py中的run_greedy，按序找到：
初始化独立状态 → Prefill/Decode调用 → argmax → 转Python整数 → 停止检查 → 下一步输入。

CPU：
```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_reference_runtime week3.test_minimal_generate -v
```

GPU命令与资源选择见 [Lesson 02第9节](LESSON_02.md)。输出文件使用新名字，拒绝覆盖旧实验。
