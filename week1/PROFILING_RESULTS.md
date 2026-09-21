# Week 1 实测解读：固定开销究竟在哪里？

日期：2026-09-21。环境：物理 GPU 2（NVIDIA L40），Python nanovllm 环境，PyTorch 2.5.1+cu124，FP32，B=8、C=64、H=4，matmul TF32 关闭，CPU intra-op 线程数 88。

本次没有修改 `mha.py` 的计算。增加了实验脚本、教程、测试，并纠正 `verify.py` 中未经验证就断言 launch 瓶颈的提示。

## 1. 数据在哪里

完整五种序列长度实验（含三个长度、四种版本的共 12 份 trace）：

`/home/ubuntu/infer-engine/results/week1/20260921T080243_917901Z/`

独立复测（整个进程不启用 profiler，T=1、64、1024）：

`/home/ubuntu/infer-engine/results/week1/20260921T080326_911531Z/`

每项有 10 次预热、每轮 100 次 forward、5 轮重复。下表取独立复测的 Wall 中位数，单位 ms/forward。精确值及各轮样本在对应 `summary.json`；计算加速比时应使用原始精度。

| T | eager | cached_mask | SDPA | CUDA Graph |
|---|---|---|---|---|
| 1 | 0.1241 | 0.1107 | 0.0709 | 0.0188 |
| 64 | 0.1540 | 0.1409 | 0.0716 | 0.0430 |
| 1024 | 2.1260 | 2.1212 | 0.2777 | 2.1229 |

这是相同输入语义、权重、精度下的控制实验；Graph 捕获成本不在稳态计时内，且固定形状/地址。

## 2. 第一条证据：T=64 的 CPU 提交与 Wall 很接近

独立复测中，eager 的 CPU 提交约 0.1537 ms，Wall 约 0.1540 ms。提交完最后一批工作后，尾部等待已很小。

这支持 CPU/框架提交路径在该小尺寸负载下限制了推进速度。不能进一步把 0.1537 ms 全称为 `cudaLaunchKernel` 开销：Python、框架 dispatcher、张量元数据处理、内存管理、runtime 调用和可能的内部阻塞都包括在里面。

## 3. 第二条证据：保留计算，用 Graph 改变提交路径

T=64 的 CUDA Graph 将 Wall 从约 0.1540 ms 降至 0.0430 ms，约 3.58 倍。

在独立的 profiler 采集中，eager 每个 forward 有 15 个 kernel，Graph 有 17 个；核心矩阵计算仍存在，并没有因为 Graph 自动被融合成一个 kernel。

本次 trace 内，5 个 forward 的 eager 有 75 次 `cudaLaunchKernel` 调用；Graph 有 5 次 `cudaGraphLaunch` 和 10 次 `cudaLaunchKernel` 调用。Graph 路径可以有辅助操作，所以“GPU kernel 数变少”不是这个实验的前提。

**支持的结论：短序列有显著的逐操作提交/调度开销，可以通过 Graph 路径减少。**

**不支持的结论：所有差值都是某一个 launch API 的时间，或者算子数学计算快了 3.58 倍。** Graph 也影响分配复用与调度方式。

## 4. 第三条证据：缓存 mask 有较小收益

T=64 的 cached_mask 约 0.1409 ms，相对 eager 约减少 8.5% 的 Wall。采集期 kernel 数从 15 降至 13，符合移出 mask 构造相关操作的变化。

这说明 mask 构造是一个可优化点，但仅移出它无法解释或消除大部分短序列开销。`masked_fill` 仍执行，也仍处理完整 scores。

## 5. 第四条证据：T=1024 的瓶颈表现不同

T=1024 的 eager 与 Graph 分别约 2.1260 和 2.1229 ms，收益约 0.15%，不足以作为稳定性能提升来宣传。Graph 的 CPU 提交已经很短，但 GPU 的工作仍然需要完成。

此时 SDPA 约 0.2777 ms，相对 eager 约快 7.66 倍。这支持优先优化 Attention 的执行方式和中间数据处理。

在大尺寸 eager trace 中，主要 GPU 时间出现在 softmax、mask、逐元素操作与矩阵乘法。原实现显式处理 `[8,4,1024,1024]` 的大张量，单个 FP32 scores/weights 张量约 128 MiB。T=64 时相同形式张量约 0.5 MiB。

这解释了为什么“大量临时数据处理”值得关注，但本次并没有采集硬件带宽/SM 利用率计数器，所以不能进一步证明“纯带宽瓶颈”或“纯算力瓶颈”。

本次 SDPA trace 的 kernel 名含 `fmha_cutlassF_f32...PyTorchMemEffAttention`，表明走的是 memory-efficient attention 路径，不能把该结果写成“FlashAttention 提速”。

## 6. 两轮绝对数字不同，怎么面对？

完整实验的 T=64 eager/Graph 为约 0.2298/0.0439 ms，独立复测为约 0.1540/0.0430 ms。Graph 改善短序列、长序列 Graph 收益很小的方向一致，但跨进程的主机侧时间存在明显变化。

完整实验在每个 T 的无 profiler 计时结束后采集 trace，再进入下一个 T；独立复测整个进程都不启用 profiler。两种运行历史不同，且服务器上其他 GPU 有任务。当前证据不能精确归因跨轮差异；可能的 CPU 调度、频率、缓存/运行状态和采集影响需要独立控制实验。

因此，参考独立复测填写性能结论，附上原始样本，不拿第一次较大的加速比作为唯一指标。要发表或写简历，最好在更稳定的时段做独立多轮复测或交替运行顺序。

Profiler 中的大空档也不能直接换算为非采集期的“空闲百分比”。它本身会产生观测开销。

## 7. 下一次请你亲自完成

1. 先看 `week1/PROFILING.md`，写下四版本的速度预测。
2. 用独立复测目录中的 `report.md` 核对预测。
3. 打开完整实验中的 `T64_eager.trace.json` 与 `T64_cuda_graph.trace.json`，找出从逐次 launch 到 graph replay 的变化。
4. 打开 `T1024_eager.operators.txt`，把前三种 GPU kernel 对应到 `mha.py` 的代码步骤。
5. 回答：为什么 Graph 在 T=64 有效，在 T=1024 几乎无效？为什么缓存 mask 不等于消除了 mask 的所有开销？

如果能用数据与代码回答这两个问题，你就已经完成了一次真正的瓶颈诊断，而不只是运行 profiler。
