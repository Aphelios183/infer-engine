# Chunk 大小与 Decode 干扰实验（2026-10-11）

> 清理说明（2026-10-11）：本报告保留关键指标与结论；下文提及的原始 trace、逐次测量 JSON 已按用户要求删除，路径仅作历史记录。复现脚本与旧实现快照仍保留，重新运行请使用新的输出路径。

## 本课结论

已完成无 profiler 的固定负载实验（6 组，每组预热 1 次，正式 5 次，共 30 个正式 trial）。当前实现中，chunk 变小没有带来稳定的尾延迟改善，反而增加分段固定开销与插入次数。不能把 Chunked Prefill 等同于性能优化成功。

生产 HybridScheduler 未改。其 decode-priority 策略本来不会在 A Decode 期间服务 B，因此另设脚本内实验策略，不能混淆两者。

## 负载与控制变量

- 单卡：物理 GPU 6，L40，开始/结束均 291 MiB、0%；执行期间一次快照 9196 MiB、58%。其他 GPU 有用户任务，机器并非独占，未锁频，不能排除共享主机调度波动。
- 模型：本地 Qwen3.5-4B，原 BF16/reference-eager 路线；不启用 CUDA Graph/TP/混合批次。
- A：固定 64-token prompt，输出 48 token；B：固定 512-token prompt，输出 1 token，以隔离 B Prefill 的干扰，不引入 B 后续 Decode。
- A 输出第 4 个 token 后提交 B。是事件驱动到达，不是固定墙钟时刻，也不是持续到达压力测试。
- 固定输入 ID 和预定输出 ID 全部记录在 JSON。真实 forward、LM Head、argmax、同步全部执行；实验在 Sequence.append_token 时用同一份 fixture 替换选出值，使后续 Decode 输入完全一致。ignore_eos=True、长度固定。
- 这是 teacher-forced 调度实验，不是正常自由生成，不用于质量结论，也没有解决前一课的 chunk 数值差异。
- max_model_len=640，max_num_seqs=2，block_size=16，总 token budget=512；当前首次准入仍为完整 prompt 预留 KV 块。
- 使用真实 GPUExecutionRunner 与实际资源池。所有 trial 验证输出 fixture、结束资源回收、没有混合 A/B 批次。
- 每种配置独立预热一次，5 轮中用固定随机种子打乱配置顺序；记录所有事件和原始 ITL，不只保存平均数。
- 不开 profiler、无中间张量 snapshot；计时利用实际 Runner 完成后的主机时间，其内部已同步。这是进程内结果可用时刻，不含网络/客户端流式传输延迟。

## 策略与对照

1. alone：只有 A，测背景 Decode。
2. decode_priority：原调度器，B chunk=128；B 必须等 A 完成，验证饥饿风险。
3. interleave_whole/256/128/64：统一实验策略，B 到达后，每累计 4 轮 Decode 给 Prefill 一轮机会。

脚本通过临时让 reserve_decode 返回 False 来使原 schedule 进入 Prefill 分支；不移走 running 队列、不绕过状态所有权、不修改生产源码。实验断言该轮确实为 Prefill。本方法只服务本次有足够容量的双请求试验，不是完整生产公平调度实现。

在四个 interleave 组中只改 chunk 大小；alone 与 decode_priority 是额外控制组，不能称为相同调度策略下的单变量比较。

## 指标定义

- ITL：A 相邻输出就绪时刻之差，共 47 个间隔；包含中间插入的 B 工作和调度。
- TTFT(B)：从提交 B 前的计时点，到 B 唯一输出就绪；包含排队与全部 chunk。
- 吞吐：总输出 token / 从提交 A 到所有请求完成的墙钟时间；双请求为 49 token，alone 为 48，alone 的工作量不同。
- 百分位：对每个 trial 的 47 个间隔线性插值计算，再取 5 次 trial 指标的中位数。不是大规模服务 P99，也不是把所有样本池化后的 P99。

## 结果（5 次 trial 指标的中位数）

| 方案 | A 平均 ITL ms | A P95 ms | A P99 ms | A 最大 ITL ms | B TTFT ms | 输出 token/s |
|---|---:|---:|---:|---:|---:|---:|
| 只有 A | 33.74 | 37.01 | 41.95 | 42.39 | — | 28.01 |
| 原 Decode 优先，B chunk128 | 37.37 | 41.77 | 48.37 | 49.79 | 2178.26 | 20.01 |
| 插入整段 512 | 40.54 | 41.78 | 132.17 | 205.67 | 315.72 | 23.59 |
| 插入 chunk256 | 43.68 | 55.20 | 184.08 | 186.27 | 601.38 | 22.44 |
| 插入 chunk128 | 49.62 | 171.67 | 193.28 | 200.13 | 1183.92 | 19.86 |
| 插入 chunk64 | 61.39 | 169.65 | 198.05 | 216.75 | 2297.63 | 16.24 |

波动不能忽略：

| 方案 | 每轮最大 ITL 的 min–max ms | B TTFT min–max ms |
|---|---:|---:|
| 原 Decode 优先 | 42.60–56.22 | 1923.74–2243.85 |
| 整段 512 | 188.44–214.61 | 284.74–326.53 |
| chunk256 | 161.12–261.27 | 518.20–658.35 |
| chunk128 | 176.07–210.29 | 1138.54–1218.29 |
| chunk64 | 148.39–227.16 | 1997.50–2392.43 |

因此不能根据 256 的中位最大间隔小于 512，宣称已找到稳定最优值。

## 事件记录如何解释结果

所有正式 trial 均满足：
- decode_priority：B 在 A 完成前获得 0 轮 Prefill。
- interleave_whole/256/128/64：B 在 A 完成前分别获得 1/2/4/8 轮 Prefill。
- A/B 不在同一批次，是轮次交替，不是同时运行 kernel。

B Prefill 轮次主机耗时（所有重复的事件平均，含 schedule/prepare/forward/最终轮采样与回收）：

| chunk | 平均每轮 ms | 轮数/请求 | 每请求 Prefill 轮次耗时合计均值 ms |
|---|---:|---:|---:|
| 512 | 167.45 | 1 | 167.45 |
| 256 | 150.68 | 2 | 301.36 |
| 128 | 140.16 | 4 | 560.63 |
| 64 | 130.91 | 8 | 1047.30 |

工作量从 512 减到 64，并没有让每轮耗时降至八分之一。当前 eager/reference 路线每段重复整套层调用，固定开销显著；具体各部分归因仍需专门 profiler 对照，不能仅由这张表认定都是 Python。

B TTFT 还包含每个 chunk 之前 4 轮 A Decode 的等待。64 有 8 次机会，累计等待更多；这是当前“固定轮数公平策略”的代价，不是 chunking 必然的通用结论。

## 为什么 P95 会误导

A 共 47 个间隔。整段只插入一次，约占 2.1%，所以一次约 206 ms 的大卡顿，P95 仍可仅约 42 ms。64 插入八次，约占 17%，会显著进入 P95/P99。

这说明：
- 平均值隐藏少数大尖峰；
- P95 也可能漏掉低频尖峰；
- 小 chunk 可能产生更多中等尖峰；
- 应同时查看 max、P95/P99、尖峰次数与事件顺序。
- 47 个间隔的 P99 只是短实验描述，不能当作线上稳定尾延迟估计。

## 可复现入口与产物

服务器：
- /home/ubuntu/infer-engine/nanovllm_qwen35/benchmark_chunk_interference.py
- /home/ubuntu/infer-engine/nanovllm_qwen35/chunk_interference_20261011.json

在 nanovllm_qwen35 目录，核实空闲卡后：
```bash
CUDA_VISIBLE_DEVICES=6 /home/ubuntu/enter/envs/nanovllm/bin/python benchmark_chunk_interference.py --output chunk_interference_new_run.json
```
输出路径必须不存在。约数分钟；保留真实模型路径与环境。不修改生产调度器，不安装依赖、不提交 Git。
脚本 SHA256：70d938e71e3130aa365ab64c435a94e85596242ed5be825ffa2ff3be681584b3
生产 scheduler.py 运行前后 SHA256：34303034c2d73edb5d6ec5fbe51dad8e2691a792e9f8d54641c3eee96842fb76

## 教学验收与下一步

先独立回答：
1. 为什么整段组的 P95 约 42 ms，却仍有约 206 ms 的最大间隔？
2. 为什么 chunk64 的单轮平均耗时较小，但 B 的 TTFT 更大？
3. 原 Decode 优先下，A 很流畅是否代表调度公平？
4. 这份结果能否证明生产 vLLM 的 chunk64 也更慢？不能：实现、批次组织、负载均不同。

下一步若继续实验，固定 chunk，再单独改变“几轮 Decode 后给 Prefill 一次机会”，不要同时改两个旋钮。更完整服务评估还需持续到达、多长短请求、更多重复、SLO 和更高效执行后端。本轮没有生产合入任何公平策略，也没有据此决定最佳 chunk。
