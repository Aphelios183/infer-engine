# 分页 Decode 性能实验：局部改善，端到端收益不稳定

实验日期2026-10-14。生产实现未在本轮修改，默认仍是dense。2026-10-15按用户要求删除实验脚本、原始JSON及两份trace，仅保留本文关键指标和实验条件。

## 协议与证据边界

- Qwen3.5-4B，单张L40物理GPU6，Torch2.5.1+cu124；BF16、TF32关闭、CPU线程2。
- GPU6开始约291MiB/0%，只有常驻243MiB进程；同机其他GPU有任务。不是独占环境、未锁频、未持续采集GPU负载。一次中间观察为53%利用率/2490MHz，不能证明全程稳定。
- 同一模型实例切换dense/flash_paged；输入长度64/256、batch1/2、输出16token，greedy且ignore_eos=True；每种路径每组预热2次，6对AB/BA交错。
- 每次创建新请求和历史状态；模型加载、分词、编译预热不计时。计时不开profiler、capture_logits或快照；保留既有同步/安全检查。TTFT含入队至首token结算，TPOT为首末token时间差除以15，总输出吞吐为B×16/整批时间。
- 整轮峰值allocated/reserved含权重、池、Prefill与Decode，不等于Decode临时张量峰值。allocator保持预热状态。
- 24对A/B、全部生成token序列一致，但没有本轮逐层output/state对拍；数值证据沿用PAGED_DECODE.md中的有限用例，不把token一致当无损证明。

## 无profiler结果

数值为6次中位数；TPOT下降为正表示改善。

| 输入/B | dense TPOT ms | paged TPOT ms | TPOT下降 | dense/paged 输出token/s | 成对TPOT改善次数 |
|---|---:|---:|---:|---|---:|
| 64/1 | 37.685 | 36.102 | +4.20% | 22.763 / 23.548 | 5/6 |
| 64/2 | 66.423 | 69.307 | -4.34% | 25.913 / 24.209 | 2/6 |
| 256/1 | 35.461 | 36.878 | -3.99% | 23.716 / 23.032 | 4/6 |
| 256/2 | 79.124 | 74.776 | +5.49% | 21.135 / 22.254 | 4/6 |

| 输入/B | dense TPOT范围 ms | paged TPOT范围 ms | dense/paged TTFT中位数 ms | 两路径相同峰值allocated GiB |
|---|---|---|---|---:|
| 64/1 | 33.410–40.312 | 32.501–39.263 | 137.044 / 132.938 | 8.04494 |
| 64/2 | 65.862–69.547 | 63.231–75.934 | 238.072 / 282.235 | 8.04524 |
| 256/1 | 33.180–39.698 | 31.833–41.238 | 142.645 / 136.515 | 8.08413 |
| 256/2 | 68.323–82.847 | 67.090–84.595 | 322.195 / 302.909 | 8.08427 |

Prefill路径未改，TTFT仍有明显波动，提示环境/运行状态影响；不能归因于分页实现的首token优化。256/B1多数配对改善但两个中位数之比退化，两种统计不是同一量，均保留，不挑有利指标。

## 独立Decode profiling

L256/B1，分别建立新历史，Prefill在trace之外，每条路径记录2步Decode；输出均为[264,1328,869]。不是性能计时样本。

选取key_averages中CPU范围（cpu_total_us>0）关联的CUDA时间；同名device-only annotation跨度含空隙，不能混作kernel耗时或累加。原JSON曾保留两类条目，现已清理。

| 指标，均为两步合计 | dense | paged |
|---|---:|---:|
| Full Attention调用 | 16 | 16 |
| Full Attention关联CUDA时间 ms | 3.680 | 3.189 |
| GDN关联CUDA时间 ms | 11.223 | 11.213 |
| 整步关联CUDA时间 ms | 34.360 | 33.690 |
| aten::index_select调用 | 36 | 4 |
| aten::cat调用 | 274 | 242 |
| aten::repeat_interleave调用 | 128 | 96 |

减少32次index_select=2步×8层×K/V；剩余调用不是历史KV gather。cat减少32次对应Full历史K/V拼接；repeat_interleave也减少32次，对应GQA不再显式展开KV头。其他层仍使用cat/repeat，不应要求全部消失。

Full Attention约下降13.34%，平均每步节省约0.245ms。该范围包含投影、Norm、RoPE、gate/o_proj；新路径还在范围内写KV，旧路径写KV在外，所以它不是严格等范围的裸Attention kernel速度比较。历史gather也位于旧full_attention函数外。没有报告FlashAttention裸kernel加速倍数。

## 怎么解释

1. 确定结论：消除了历史gather、拼接与显式KV头扩展；短trace里局部关联CUDA时间下降。
2. 当前不能确认：稳定TPOT/吞吐加速、整轮峰值显存下降或任务质量无损。四组计时有升有降，范围重叠。
3. 局部节省很小，而GDN、MLP、逐请求执行、同步和主机开销仍在；Amdahl效应可能限制收益。不能把所有波动直接归因于Python，也不能把profiler下的时间比当端到端提速。
4. 全轮峰值不变不等于gather没有临时开销；后续需单独重置Decode峰值、去除Prefill峰值干扰才能回答。

## 下一步

保留显式实验选项，不切默认。若继续分页性能实验：扩大配对样本、控制共享负载、测Decode专属峰值及更长上下文；数值方面独立验证真实层Q/K/V和长生成。后续LM Head批量化结果见LM_HEAD_BATCH_20261014.md，不与本实验合并计算收益。

清理范围：benchmark_paged_decode.py、paged_benchmark_20261014/results.json、dense.trace.json、flash_paged.trace.json，以及对应本地结果/补丁副本。原始逐次样本和trace已不保留，无法重新统计或回看此次时间线；若复测须按本文协议重建入口。删除项原为Git未跟踪文件，不能用Git恢复。生产分页路径和课程文档保留，未自动stage/commit。
