# Lesson 4：Qwen3.5-4B 真实数值诊断报告

会话日期：2026-09-29。结论：真实实验已运行，整体 status=diagnostic、passed=false；正式验收未完成。

## 执行条件

物理 GPU5，UUID GPU-093b8773-f073-4c18-87c7-2afa616fd9c2，NVIDIA L40；运行前后均约289 MiB、利用率0%，不是独占卡。
模型固定 ModelScope revision ed182e32090db791077e12e0f58d22f3daafa173，加载前执行 Git/LFS 校验。
Python 3.11.15、torch 2.5.1+cu124、transformers 5.8.0，原 nanovllm 环境未改依赖。
Transformers 执行模型 forward，自己的循环管理输入、位置、混合缓存与生成；不是 nano-vLLM 原生执行。
BF16、eval/inference_mode、Full Attention eager、Linear Attention 使用已有 torch 回退。TF32 关闭。
BF16 与 FP32 校准模型顺序加载，非同时驻留。完整模型包含未使用的视觉权重，但只测试纯文本。

## 1. 固定相同前缀的对拍

每个用例比较 Prefill + 四次固定 Decode，即五个比较点。
两边每一步对应相同 token 前缀；全重算从空状态处理整个前缀，缓存路径只输入新 token。
固定续接来自“缓存复用历史信息。”编码的前四个 token，不根据两条路径各自 argmax 推进。
所有 logits 有限，比较整个词表的最后位置 logits。

| 用例 | 已处理长度 | max_abs 最大值 | mean_abs 最大值 | relative_l2 最大值 | top1 不同次数 |
| --- | --- | --- | --- | --- | --- |
| 用一句话解释 KV Cache。 | 18→22 | 0.25 | 0.02336352 | 0.01307054 | 0/5 |
| 计算 2+3，只输出结果。 | 22→26 | 0.1875 | 0.02831743 | 0.01520138 | 0/5 |
| What is prefill in language model inference? | 21→25 | 0.1875 | 0.02804025 | 0.01442282 | 0/5 |
| T=1 普通原始 token | 1→5 | 0.1484375 | 0.02181847 | 0.01240929 | 0/5 |

四个 Prefill 比较点误差均为0。Decode 存在非零差异；全部20点 top1 一致，不等于完整 logits 数值验收已经通过。
表中三个最大值可能来自不同步骤，逐步记录见原始 JSON。

## 2. 请求隔离与异常恢复

复用同一个模型实例，比较 B=“你好”单独运行与两种前置情形：

- A 正常生成结束，再运行 B。
- A 的 Prefill 已成功返回缓存，随后 Decode 更新状态后主动抛出指定异常，再运行 B。

B 首先直接经过与 A 相同的 run_greedy 入口，再做固定前缀补充检查；避免先用另一个 B 请求重置状态后才测试生成入口。
故障发生在 Decode、position_start=18，本次输入为 token 79852；生成循环已经持有非空历史缓存，不是只在 Prefill 返回前出错。

| 场景 | B 固定前缀 logits 最大差异（5点） | B 生成路径 logits 最大差异（4点） | 输出与停止行为 |
| --- | --- | --- | --- |
| A 正常结束后 B | 0 | 0 | 与 B 基线一致 |
| A Decode 异常后 B | 0 | 0 | 与 B 基线一致 |

B 生成 IDs 为 [109266,6115,98691,111454]；三次一致。各次从位置0、自己的输入长度开始。
这些用例未观察到请求数值串扰；不覆盖全部异常类型、长期压力运行、并发和显存泄漏。
统一报告仍为 diagnostic，因为未加载批准的容差文件；精确零差异是观测事实，不借此把整体对拍升级为 pass。

## 3. 为什么还不能正式通过

独立校准的 BF16 重复运行误差为0；BF16 对 FP32 最大差异为0.32021713。
原计划四倍公式给出 max_abs 上限1.28086853，而校准最小 FP32 top1/top2 margin 只有0.02321339。
审阅在验收提示词诊断运行前拒绝了这个过宽候选值，approved=false 保持不变。
完整依据见[容差审阅](LESSON_04_CALIBRATION_REVIEW.md)。
本次数据不足以将 Decode 差异全部归因于 BF16 舍入，也不足以认定缓存实现错误。
下一步应先在独立样本上区分精度变化与增量执行顺序的影响，重新设计并冻结标准；不得从本表倒推“刚好能通过”的门槛。
校准代码早于隔离增强，代码指纹不同；旧未批准文件也不能直接用于新代码正式验收。

## 4. 工具测试与审阅

- 检查器最初5项测试先失败后通过；隔离审阅修复增加1项回归。
- 最终课程回归：77项，75通过、2项默认跳过。跳过不算通过。
- 只读审阅发现的 Important 问题已修复：异常注入改为已有缓存的 Decode，B 经过真实生成入口；保存正常与异常路径证据。
- 最初实跑因可选 generation_config.json 不存在而失败，修正为明确记录 null；没有增改模型文件。错误日志保留。
- 本轮不声称 Week 3 数值验收完成，不报告吞吐优化，不修改用户 reference_runtime.py 的手写注释。

## 5. 代码、证据与复现

- [验证器](verify_generation.py)、[CPU测试](test_verify_generation.py)。
- [校准原始JSON](results/qwen35-lesson4-calibration.json)、[未批准容差](results/qwen35-lesson4-calibration-tolerances.json)。
- [首轮诊断](results/qwen35-lesson4-diagnostic.json)保留作历史，不作为增强后的异常恢复证据。
- [最终诊断JSON](results/qwen35-lesson4-diagnostic-v2.json)。
- [执行记录](../archive/development/lesson4-progress.md)、[日志目录](../archive/development/lesson4-validation/)。

最终验证器 SHA256：3b733588bc5c946125538fd3f818626928b7a49f83ec2c2fe43b8cfd770772f0。
每份JSON含模型、环境、输入与代码指纹。结果路径已存在时拒绝覆盖。

复跑前检查 GPU 并选择合适编号，不沿用历史空闲状态。以下输出名称须保持未使用：

```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES=<当时选定GPU> HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.verify_generation --verify --output week3/results/qwen35-lesson4-new-diagnostic.json
```

未传 --tolerance-file 时只能 diagnostic。正式 --verify 会拒绝未批准、指纹不匹配或非法的容差。
