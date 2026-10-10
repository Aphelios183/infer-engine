# 小课3：positions与slot_mapping轮内复用

## 本次只改了什么

学生正确预测单请求Decode H2D会从50降为12。本次在 `qwen35_adapter/pool_backend.py::PooledCPUBackend.forward` 的层循环前增加：

```python
positions_gpu = inputs.positions.to(self.device)
slots_gpu = inputs.slot_mapping.long().to(self.device)
```

层内分别使用 `positions_gpu[a:b].unsqueeze(0)` 与 `slots_gpu[a:b]`，不再逐层上传。GPU backend继承此实现，因此真实GPU路径生效。

两张张量的内容只读，生命周期只属于当前forward；没有把它们存成跨批共享缓冲，没有增加stream、non_blocking或异步复用。现有同步、输入校验、状态所有权、历史地址构造和缓存写回均保留。

slot_mapping没有层信息：先用compact层索引选择KV池切片，再用slots选择本层token地址，因此跨层共享索引不会互相覆盖。

## 三层验证

### 1. 正确性

GPU2真实4B模型原有验证入口通过：长度1/3/19/63/64/65/128/257及混合长度[63,65]，共27步logits与HF参考精确相等；Full KV、conv/recurrent池精确相等。含双请求重排、跨块边界、强制EOS、零预算和forward之后主机失败回收验证。没有新建CPU测试目录。

此外A/B实验每对32-token生成输出均一致。不能据此声称所有模型、长度、dtype和多流场景都已验证。

### 2. 独立profiler：确实减少了重复拷贝

L=64，Batch1/2，每组1次Prefill+2次Decode，插桩前后输出一致。

| 每步 | 旧H2D | 新H2D | 旧cudaStreamSynchronize | 新cudaStreamSynchronize |
| --- | --- | --- | --- | --- |
| B1 Prefill | 42 | 4 | 84 | 46 |
| B1 Decode | 50 | 12 | 92 | 54 |
| B2 Prefill | 82 | 4 | 164 | 86 |
| B2 Decode | 98 | 20 | 180 | 102 |

单请求Decode：1 input_ids + 1 positions + 1 slots + 8历史addresses + 1 last_indices = 12。
双请求Decode：整批4次公共上传 + 16次历史addresses = 20。

每步Pinned DtoH仍为41/81次，表明逐层布尔校验没被删除。整组kernel数量仍为19235/38515，说明模型计算量和算子数量没有减少。我们优化的是主机侧重复元数据传输，不是实现了批量层计算或融合算子。

### 3. 无profiler交错A/B：初步性能收益

2026-10-10，GPU2 L40，测前/后295MiB、0%利用率，仍有常驻占用；不宣称独占机器。PyTorch环境与前次相同。L=64、生成32token、max_model_len=512、max_num_seqs=2、block_size=256，greedy、ignore_eos=True、TF32关闭、CPU线程2。只加载一份模型，旧forward和新forward在同一backend上替换调用，每种路径预热2次，6对交错AB/BA，采样期间不开profiler/快照。

| 指标中位数 | B1旧 | B1新 | B2旧 | B2新 |
| --- | --- | --- | --- | --- |
| TTFT ms | 120.36 | 125.49 | 236.79 | 235.80 |
| TPOT ms | 38.62 | 36.21 | 73.84 | 67.38 |
| 总输出token/s | 24.29 | 25.59 | 25.32 | 27.48 |
| 输出token/s范围 | 20.98–25.02 | 22.88–26.77 | 24.42–25.87 | 26.79–27.97 |

按吞吐中位数之比，B1提高约5.34%，B2提高约8.53%；TPOT分别降低约6.23%和8.75%。B1六对吞吐变化为-4.28%、+9.06%、+5.88%、+8.79%、+4.80%、+10.91%，有一对退化；B2六对均为正，约+3.55%至+11.16%。TTFT没有稳定改善，不宣传首token加速。

峰值allocated：B1旧/新均8638183936字节；B2旧8638511616、新8638512640字节（+1024字节）。并非显存优化。样本量小且共享环境，本结论限于该负载；后续扩大上下文/批次前需另测。禁止拿带profiler耗时或昨天的基线直接计算本次收益。

## 产物与复现

服务器适配根目录 `/home/ubuntu/infer-engine/nanovllm_qwen35`：

- `metadata_reuse_correctness_20261010.json`：真实模型验证结果。
- `metadata_reuse_ab_20261010.json`：保留小型逐次A/B证据；临时GPU采样CSV已清理。
- 旧/新profiling目录下的trace和生成summary已清理，关键计数保留在上表；不能再回看这次原始时间线，可重新运行脚本采集。
- `benchmark_metadata_ab.py`：交错实验。当前固定输出名，重跑前需使用新的输出名，不能覆盖原始数据。
- 旧forward归档 `/home/ubuntu/infer-engine/archive/metadata-reuse-20261010/pool_backend_before.py`，仅供对照，生产入口使用新实现。

旧源码SHA256：`d911ad0bea71f0d22c3e0dac10f55f86fb60267510d2326c47dd0d0b5bb14033`。
新源码SHA256：`8399c28a847d443f124c563f14f98cb5bf5dd53e462548f613b13a4b1cc4c28e`。

本轮完成一个小优化闭环；不代表六阶段中的“元数据/同步”全部完成，也不自动认定学生已独立实现。下一道结构题：同一请求本轮8个Full层的历史addresses也相同，为什么它们仍被构造和上传8次？如果下一步复用，应按整批还是逐请求准备，为什么？

2026-10-10课程推进：学生已理解历史addresses按请求区分、同轮跨Full层可复用及past=0的空历史边界。历史地址复用尚未实施，Decode 5/6次H2D仍是理论目标，不能写成已实现。先暂停小拷贝优化，进入小课4的分页存储与执行接口学习。
