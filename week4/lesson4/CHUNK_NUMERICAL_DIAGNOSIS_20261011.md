# Chunked Prefill 首个数值分歧定位（2026-10-11）

> 清理说明（2026-10-11）：本报告保留关键指标与结论；下文提及的原始 trace、逐次测量 JSON 已按用户要求删除，路径仅作历史记录。复现脚本与旧实现快照仍保留，重新运行请使用新的输出路径。

## 结论与边界

在物理 GPU 6（L40，运行前 291 MiB / 0%）复现固定 65-token 输入，比较整段 65 与分块 64+1。已定位最早观测到的算子级差异：**第 0 层 GDN 的 in_proj_qkv 线性投影**。输入及 RMSNorm 输出一致，但 BF16 投影的 M=65 与 M=64/1 计算结果不完全一致。

不是第二段才开始出错，也不是“整段可看到未来，分块看不到”。因果可见范围应相同。此次证据支持形状相关投影数值差异、分段 recurrent 状态舍入及后续传播共同参与；不能把最终全部误差定量归因于一个因素，也不能据此宣布质量合格或排除所有其他问题。

生产模型、资源池 dtype、调度实现均未修改。仅新增诊断脚本；FP32 消融只在诊断进程临时开启并恢复。

## 可复现入口

服务器代码：/home/ubuntu/infer-engine/nanovllm_qwen35/diagnose_chunk_boundary.py

在 nanovllm_qwen35 目录，先核实空闲卡，使用：
```bash
CUDA_VISIBLE_DEVICES=6 /home/ubuntu/enter/envs/nanovllm/bin/python diagnose_chunk_boundary.py --output profiling_chunk_new_run
```
输出目录必须不存在。需已安装环境、实际模型；不下载依赖，不更改权重。
最终结果：profiling_chunk_final_20261011/report.json 与 trace.json。
早期诊断：profiling_chunk_boundary_20261011、profiling_chunk_ablation_20261011，非最终验收数据。

Torch 2.5.1+cu124；TF32 关闭；固定 tokenizer 输入 ID 记录于 JSON；BF16 模型与原有 recurrent 池。独立 HF chunk 函数来自服务器已安装版本，不宣称其为未修改的最新上游。
layer_compute.py SHA256: 57c79a276f2cbc24eb5479ec1d127f62b6fb783593747ce317f447c03c90b73d
pool_backend.py SHA256: 8399c28a847d443f124c563f14f98cb5bf5dd53e462548f613b13a4b1cc4c28e
诊断脚本 SHA256: 2d8bc93d69055631d656e3484320b4a3bf2d0f96cf1da5b8a61c55d88de7c693

## 实验设计（对应代码）

1. run(): 使用真实 Engine/Scheduler/Runner/池，prompt 固定，只产生一个输出 token，不进行自回归 Decode。最后仍采样以正常结算，但比较的是采样前层输出/logits，不受选出的 token 反馈影响。
2. hooked(): 包装 pool_backend.decoder_layer，采集输入、绝对 positions、进入该层的历史状态、输出和返回状态。快照是诊断开销，不算模型缓存。
3. 分别运行整段 65、分块 64+1、独立前缀 64。逐层比较第一段、尾 token、边界状态，以及第二段从真实池读回的状态。
4. isolated_layer0: 固定第 0 层归一化输入，单独比较投影与 GDN，避免上游误差混入。
5. core_same_inputs: 固定 q/k/v/g/beta，单独改变 chunk/recurrent 计算方式与 FP32/BF16 边界状态。
6. ablations: 全模型分别临时使用 FP32 recurrent 池、FP32 线性投影后转回原 dtype、两者同时。不是模型全 FP32，LM Head、Attention matmul 等未统一升精度。
7. profiling: 另行关闭所有 snapshot/capture_logits，分别预热整段和分块，再独立采集两段 CPU/CUDA 时间线。禁止用数值快照阶段计时判断性能。

## 边界与层级结果

- 32 层第一段输出与独立 64-token 前缀输出逐元素相同。
- 8 个 Full 层的 KV、24 个 GDN 层 conv/recurrent：第一段返回状态与独立前缀状态相同。
- 第二段从池中读取的所有历史状态，与第一段返回状态逐元素相同。可排查本例的写回/读取破坏。
- 所有层第二段 positions 都是 [64]，不是物理地址，也没有重置为 0。
- 第一段与整段 65 的前 64 个位置并非逐元素相同。独立前缀对齐不等于整段内部的状态边界已数值相同。

| 观测位置 | 最大绝对误差 |
|---|---:|
| 第 0 层输入 RMSNorm（前 64） | 0 |
| 第 0 层 in_proj_qkv（前 64） | 0.03125 |
| 第 0 层 in_proj_qkv（最后一个） | 0.015625 |
| 第 0 层 in_proj_z（前 64） | 0.0625 |
| 第 0 层输出（前 64） | 0.015625 |
| 第 0 层输出（最后一个） | 0.00048828125 |
| 第 7 层输出（最后一个） | 0.25 |
| 第 15 层输出（最后一个） | 1.15625 |
| 第 31 层输出（最后一个） | 1.9375 |
| 最终 logits | 2.111328125 |

误差不单调增加，残差、归一化、门控、Attention/MLP 会改变误差大小。最终 logits 相对 L2=0.174266（17.43%），不是可直接忽略的误差。

## 隔离变量与解释

固定同一组 GDN 核心输入，仅比较整段与 64+1：
- 保留 FP32 边界状态：核心输出 max_abs 5.96e-8，最终 state 约 1.5e-6。
- 边界转 BF16：核心输出 max_abs 0.00012207，最终 state 0.01182985。
- 上述尾段使用 chunk 或 recurrent，本例结果非常接近。因此“尾段换了递推算法”不是这一隔离实验的大误差来源。
- 同一输入全程 recurrent 对全程 chunk，输出仍有 0.0009765625 差异：浮点求值顺序也有影响。

整段一次 GDN 调用内部在 FP32 中跨越内部 64-token chunk 边界；外部分块会把返回 state 写回 BF16 池再读入，新增一次舍入。FP32 状态信息转 BF16 后不能靠随后转回 FP32 恢复。

全模型消融（各行比较该配置下的整段与分块，不是对原模型的精度评分）：

| 配置 | logits max_abs | 相对 L2 |
|---|---:|---:|
| 原配置 | 2.111328125 | 17.4266% |
| 仅 recurrent 池 FP32 | 0.8515625 | 5.2364% |
| 仅 lc.project 使用 FP32 计算再转回 | 0.328125 | 3.4786% |
| 两者同时 | 0.75 | 6.8099% |

消融不具有可加性，两个改动叠加反而高于只改投影。BF16 回写、其他矩阵运算和非线性传播仍在；甚至 FP32 投影再转回 BF16，QKV 跨形状还剩 max_abs 0.0078125。因此本次证明了具体贡献因素，未证明两项即可完全修复，不应直接合入 dtype 改动。

## 独立 profiling

最终 trace 按 CPU user_annotation 时间范围统计；执行含准入、调度、准备、模型层、最后一次 LM Head/采样与回收。单请求同步执行，GPU kernel 均在对应范围内。GPU annotation 跨度不是 GPU 忙碌时间；以下单独累计 kernel duration，不把两类 annotation 混算。

| 指标 | 整段 65 | 分块 64+1 |
|---|---:|---:|
| aten::linear 调用 | 249 | 497 |
| CUDA kernel 数 | 14053 | 16312 |
| H2D 次数 | 4 | 15 |
| D2H pinned 次数 | 41 | 81 |
| D2H pageable 次数 | 1 | 1 |
| D2D 次数 | 113 | 433 |
| cudaStreamSynchronize 次数 | 46 | 97 |
| kernel duration 合计 | 45.198 ms | 56.553 ms |
| 带 profiler CPU 范围 | 264.451 ms | 318.284 ms |

原因：整段 248 次层内线性运算+1 次 LM Head；分块两遍层内运算+仅最后 1 次 LM Head=497。第二段还要读取历史 KV、重复校验、同步、更新状态；不能把分块理解为减少单请求总工作。
trace 中 BF16 GEMM/GEMV 组合随形状变化；这是调用路径证据，不是 Nsight Compute 指令级归因。没有执行 NCU。

这些时间含大量 profiler 开销、只是一轮结构观察、非独占机器，不是正式 TTFT/吞吐结论。当前未混合批次，也未测并发公平性，不能推导 chunking 对在线服务无价值。

## 下一步教学与工程边界

先读 layer_compute.gated_delta_net 的 in_proj_qkv、chunk 分支与 recurrent.to(state_dtype)，再读池的 copy_。结合 report 中 layer0、history_readback 与 core_same_inputs，分别解释“算出不同”和“写回错误”。

下一步应扩大固定输入/长度覆盖，并以更高精度、固定投影求值路径继续隔离剩余差异；再制定 logits 与生成质量标准。暂不放宽容差、不宣称已修复、不默认开启 chunked prefill，不把 FP32 消融当作性能优化。
