# 带历史 Chunk Profiling 与 KV 地址复用（2026-10-11）

> 清理说明（2026-10-11）：本报告保留关键指标与结论；下文提及的原始 trace、逐次测量 JSON 已按用户要求删除，路径仅作历史记录。复现脚本与旧实现快照仍保留，重新运行请使用新的输出路径。

## 完成情况

已在物理 GPU 6（L40）完成：
1. 128-token prompt 的第二段：past=64、chunk=64、单请求，分阶段 profiling。
2. 历史 KV 地址移出层循环，按请求在每次 forward 内构造并上传一次。
3. 旧/新实现的严格数值比较、20 组交错 A/B 无 profiler 测量。
4. 10 类 CPU 控制验证、38 步真实 GPU 同分块参考对拍，以及失败/取消/复用/EOS 检查。

生产改动仅 qwen35_adapter/pool_backend.py，13 行增加、3 行删除；无调度、数学算子、精度、采样语义改动。助手完成实现与实测，不代表学生已独立掌握。

## Profiling 设计与结果

第一段未计入采样；第二段是最终 prompt chunk，因此包含一次 LM Head、采样与结束回收。相同模型、固定 token、BF16 原路径、TF32 关闭。先预热两次，独立采一轮；不抓中间张量，不启用 shape/stack/memory 记录。

通过 record_function 包装 schedule、prepare、forward、GDN、GDN_chunk_core、FullAttention、MLP_projection、LM_head、state_sync、postprocess。MLP_projection 只含三个投影，不含激活/门控/残差。prepare 标记仅覆盖 Runner 的准备，backend 内重复校验包含在 forward 中。采样与其他同步可在原始 aten/CUDA 事件中查看，不误归为 state_sync。

基线 CPU 事件关联的 CUDA 时间（不是 CUDA user annotation 跨度）：

| 范围 | 调用次数 | CPU inclusive ms | 关联 CUDA ms |
|---|---:|---:|---:|
| 整步 | 1 | 276.69 | 42.19 |
| forward | 1 | 273.57 | 40.42 |
| GDN | 24 | 222.33 | 27.59 |
| GDN chunk core（GDN 子范围） | 24 | 190.26 | 21.30 |
| MLP 三个投影 | 96 | 6.67 | 8.70 |
| Full Attention 计算 | 8 | 16.12 | 2.16 |
| LM Head | 1 | 0.18 | 1.76 |

GDN 关联 CUDA 时间占整步约 65%，其中核心本身约占整步 50%。是本 workload 的主要热点。
CPU 与 CUDA 时间可重叠，父子范围不可累加；CPU 222 ms 不能直接宣称为纯 Python 耗时。GPU annotation 长跨度含间隙，不等于 GPU 忙碌时间。正式性能用下文无 profiler 数据。

## 代码改动：只复用索引，不复用 KV 数据

旧实现每个 Full 层重复：
```python
logical = torch.arange(past)
addresses = block_table[logical // block_size] * block_size + logical % block_size
addresses = addresses.to(device)
```

新实现先为每个请求构造 history_addresses[row]，8 个 Full 层从列表取同一个索引 tensor。每层仍使用自己的 kpool/vpool 做 index_select；KV gather 形成稠密临时张量的方式未改变，尚不是高效 paged-attention kernel。

正确性边界：
- 以请求 row 和该请求 block_table 构造，不能只用长度作为缓存键。
- 每次 forward 重新构造，不能跨批次复用旧 block ownership、row 顺序和长度。
- past=0 不构造历史地址、不读取历史 KV。
- positions/slot_mapping 的既有批次内复用保留。
- 输入校验、状态所有权检查、重复 forward 防护不删除。
- 不让不同层共享 KV 数据，只让它们共享索引。
- 地址 tensor 仅存活于本次调用，异步执行仍遵循现有 CUDA stream/allocator 生命周期；未加入异步跨流机制。

## 优化效果：确定消除重复搬运，速度提升证据有限

独立 before/after trace：

| 指标 | 旧实现 | 新实现 |
|---|---:|---:|
| H2D | 12 | 5 |
| cudaStreamSynchronize | 54 | 47 |
| CUDA kernels | 13387 | 13387 |
| D2H pinned / pageable | 41 / 1 | 41 / 1 |
| D2D | 281 | 281 |
| kernel duration 合计 ms | 41.733 | 41.697 |

历史地址上传从 8 次变为 1 次，减少 7 次 H2D；其余 4 次仍为本轮其他输入/索引上传。没有减少模型计算 kernel，也没有消除 KV gather。不能据此期待大幅提速。

无 profiler 同一模型实例，旧源码通过独立命名空间载入；每个样本重新执行第一段建立新状态，防止在已更新状态上重复 benchmark。只计第二段 schedule→完成与清理；真实同步确保结果已完成。各版本预热 3 次，20 对样本交替 AB/BA。

- 旧耗时中位数：168.7417 ms。
- 新耗时中位数：166.5514 ms。
- 两个中位数的相对下降：1.298%。
- 20 对中 15 对改善；逐对改善率中位数 1.058%。
- 逐对改善率范围 -23.724%～17.189%，含明显波动。

因此只记录“小幅改善迹象，重复搬运确定减少”，不声称已证明稳定 1.3% 加速。机器共享，未锁频，不能排除 CPU 调度与运行状态变化。优化保留的理由是行为不变、去除确定冗余、没有引入跨批次缓存；不是已经解决主要瓶颈。

## 正确性验证

A/B past64+chunk64：
- 第一段 conv/recurrent：逐元素一致。
- 最终 logits、全部 128 个有效位置各层 KV、conv/recurrent：逐元素一致。

现有 verify_chunked_prefill.py 回归：
- 多长度、多请求、批次换序、不同块表、无采样中间段、单 token 尾段。
- 38 步 logits 和资源池状态均与相同分块的已安装 HF 参考逐元素一致。
- 部分执行后失败回收、取消后 slot 复用、EOS 通过。
- 整段与分块的既有数值差异仍然存在；本次优化不修复也不掩盖该问题。

## 产物与复现

服务器根：/home/ubuntu/infer-engine

- nanovllm_qwen35/qwen35_adapter/pool_backend.py：新实现。
- archive/history-reuse-20261011/pool_backend_before.py：旧实现快照，可审计恢复。
- nanovllm_qwen35/benchmark_history_reuse.py：分阶段 profiling、精确对拍、无 profiler A/B。
- nanovllm_qwen35/profiling_history_before_20261011/{report.json,before.trace.json}
- nanovllm_qwen35/profiling_history_after_20261011/{report.json,after.trace.json}
- nanovllm_qwen35/history_control_20261011.json
- nanovllm_qwen35/history_gpu_regression_20261011.json

在 nanovllm_qwen35 内核实空闲卡后：
```bash
CUDA_VISIBLE_DEVICES=6 /home/ubuntu/enter/envs/nanovllm/bin/python benchmark_history_reuse.py --before ../archive/history-reuse-20261011/pool_backend_before.py --output profiling_history_new_run
```
加 --baseline 可仅采旧路径。输出目录必须不存在。回归另用 verify_chunked_prefill.py，选择新 JSON 路径。原始 trace 已留存，无需提交大 trace（现有 profiling_* 忽略规则适用）。

旧 pool_backend SHA256: 8399c28a847d443f124c563f14f98cb5bf5dd53e462548f613b13a4b1cc4c28e
新 pool_backend SHA256: 2aa710fb35d68b31d7a28ac218b48bd81c41b1d6c3c0adea1cd3b7a15d4ba592

## 下一课

优先深入 GDN reference chunk 核心，而不是继续仅优化地址上传：
- 先解释归一化、conv 历史、g/beta、chunk 核心、状态写回。
- 检查当前 reference 分支是否还执行了只为 recurrent 分支准备的重复运算。
- 再评估融合后端的接口/依赖与数值策略；不要未经核查直接换内核或安装依赖。
- 每项改变单独执行相同分块正确性与整段差异诊断，以及无 profiler A/B。

学生验收题：两个请求的 past 都是 64，但块表不同，能否共享同一个 history_addresses？为什么同一请求的 8 个 Full 层却可以共享？
