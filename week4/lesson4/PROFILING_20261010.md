# 小课2：独立profiling实测记录

2026-10-10清理记录：关键指标保留在本文及METADATA_REUSE_20261010.md；原始CPU/CUDA trace与生成summary已按用户要求删除，未留备份。下文trace路径仅为历史记录，不能再打开本次时间线；保留profile_structure.py可重新采集。

## 范围与执行状态

2026-10-10，真实Qwen3.5-4B、物理GPU2 NVIDIA L40。执行前后均约295 MiB、0%利用率，有原有常驻占用，非独占GPU。沿用nanovllm环境，未升级服务器依赖。

脚本：`/home/ubuntu/infer-engine/nanovllm_qwen35/profile_structure.py`。每组输入64 token，Batch=1/2，生成3 token，即1次Prefill和2次Decode。每组先做2次无profiler预热；记录CPU/CUDA活动，不记录调用栈、shape或内存profile，降低诊断开销。每组插桩后输出token与前一次无profiler输出一致；这是插桩一致性检查，不是重新做HF对拍。

只在独立进程内包裹方法、为原runner函数插入record_function范围；没有修改生产engine/backend文件。六个主标记：schedule、prepare、forward、LM_Head、sampling_sync、postprocess。附加标记为每层每请求、token_ids_to_host与explicit_device_sync。

原始trace：`/home/ubuntu/infer-engine/nanovllm_qwen35/profiling_20261010/batch1.trace.json`（约48MiB）、`batch2.trace.json`（约96MiB）；当时同目录生成summary.json。本次清理后均不再保留；未上传外部网站。

## 1. 逐请求执行：已观察到，不再只是猜测

Batch=2的每轮调用顺序为layer0/request0、layer0/request1、layer1/request0、layer1/request1……，共64个层调用。整份Batch=2 trace的38515个kernel均在stream 7上。证据支持当前两个请求逐层串行提交/执行，并非将两请求的投影等操作合并为批量矩阵计算。

Batch=1共19235个kernel，Batch=2共38515个，分别覆盖整组3步。这并不说明全部kernel开销来自Python，也不说明合批就必然获得2倍收益。

## 2. H2D：搬运的数据小，但反复搬

每步实际gpu_memcpy事件数：

| 阶段 | Batch=1 H2D | Batch=2 H2D | Batch=1 stream同步 | Batch=2 stream同步 |
| --- | --- | --- | --- | --- |
| Prefill | 42 | 82 | 84 | 164 |
| Decode第1步 | 50 | 98 | 92 | 180 |
| Decode第2步 | 50 | 98 | 92 | 180 |

单请求Decode第1步：41次8字节H2D位于forward、8次512字节H2D位于forward、另1次8字节H2D位于LM Head前的输入选择。结合源码可以逐项解释：

- 1次input_ids。
- 32层分别传positions。
- 8个Full层分别传slot_mapping。
- 8个Full层分别构造并传历史addresses；此时past=64，64个int64即512字节，下一步变520字节。
- 1次last_indices。

总计1+32+8+8+1=50。双请求除input_ids与last_indices外按请求重复，因此为1+64+16+16+1=98。Prefill没有有效历史，空地址张量不产生相应的实际H2D事件。

源码入口：`qwen35_adapter/pool_backend.py::forward` 中positions、addresses、slots的 `.to(self.device)`，以及gpu_engine中last_indices传输。

这些数据量并不大，不能称为“PCIe带宽已满”。问题候选是大量小拷贝及随之出现的提交/同步开销。

## 3. DtoH：不只有最后的tolist

每个单请求步有41次1字节Pinned DtoH及1次token ID回传；双请求为81次1字节及1次16字节token ID回传。CUDA External id与CPU操作对应到：

- 每层位置验证的 `aten::equal → aten::item → aten::_local_scalar_dense`。
- Full层partial RoPE的负位置判断，经 `aten::is_nonzero → aten::item` 读取布尔值。
- 采样前logits有限性检查。
- 最后的 `ids.tolist()`。

32层位置相等检查+8层负位置检查=每请求40个布尔结果，再加每批1个logits检查。源码中这些检查具有正确性作用，不能未经替代验证直接删除。

单请求两个Decode步中，token_ids_to_host的CPU范围约0.048/0.045ms；双请求约0.083/0.051ms。本次并没有看到tolist独占数十毫秒。等待可能已经在前面的检查或拷贝处发生，因此不能据此推断取消所有同步也无收益。

## 4. 六阶段区间和GPU间隙怎样解读

以下仅为本次插桩后的Decode第1步，不能替代无profiler基线：

| CPU范围（含子调用） | Batch=1 ms | Batch=2 ms |
| --- | --- | --- |
| 整步 | 62.881 | 145.313 |
| schedule | 0.039 | 0.043 |
| prepare合计（两次） | 0.159 | 0.199 |
| forward | 60.616 | 141.204 |
| LM_Head提交范围 | 0.112 | 0.153 |
| sampling_sync | 1.871 | 3.634 |
| postprocess | 0.021 | 0.021 |

prepare出现两次是因为runner先准备，backend的_check_inputs又重建一份校验；后一次嵌套于forward。不要把上表所有行相加。LM_Head为CPU提交范围，其GPU执行可延伸到sampling_sync，不能认定矩阵乘只花0.1ms。最后一个Decode的postprocess还包含结束回收。

同两步的已采集kernel/memcpy/memset时间区间并集约17.09/34.16ms，明显小于插桩整步时间。不同Decode步中最大的内部设备活动间隙约0.48–4.00ms；相邻CPU活动包括矩阵乘提交、索引写回、复制等。不能把“整步减去GPU活动”全部叫做Python耗时：还包含profiler扰动、CPU执行/调度、同步与追踪范围限制，也不代表其他进程没有GPU活动。

尤其单请求Prefill的schedule约31.68ms，单次诊断有明显扰动，不能据此声称调度器成为性能瓶颈。

## 小课3的候选：先改批次元数据，不直接删除检查

首选是将positions、slot_mapping与历史地址在每批/每请求准备一次，并跨层复用GPU张量。原因是重复次数已由trace确认，职责边界清晰，适合一次小改动学习。

先定义输入生命周期、请求顺序与地址不变量；再替换逐层重复上传。位置合法性检查是否能提前到CPU侧并合并，是后续独立改动，需要确保语义等价。保留资源释放前的必要同步，不把所有synchronize一概删掉。

验收：输出/状态正确；重新profile看到相应H2D次数下降；在不启用profiler的交错A/B实验中判断端到端是否改善。次数减少并不保证吞吐提升。本轮只采集和分析，没有实施优化。

课堂问题：单请求Decode的50次H2D中，哪些张量在32层/8个Full层之间其实不变？你会让谁持有这些GPU张量，确保下一批不会提前覆盖？

## 复现

先nvidia-smi核实空闲卡，再在适配根目录运行，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python -u profile_structure.py --output-dir profiling_repeat
```

解析时只把CPU user_annotation中的step当作步骤，排除同名gpu_user_annotation，避免重复计算。原始trace仍保留完整CPU/CUDA时间线。
