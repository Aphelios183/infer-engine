# Week5：从 Qwen3.5 适配进入分页执行与批量化

更新：2026-10-14。指接下来的第一学习周，不是承诺七天完成。上级：[路线图](../ROADMAP.md)；下一周：[Week6](../week6/PLAN.md)。

## 起点与目标

2026-10-14进度覆盖：已接通显式flash_paged Decode，Prefill仍沿用旧路径；完成有限层级/模型冒烟及4组性能实验，见[PAGED_DECODE](PAGED_DECODE.md)和[性能报告](PAGED_BENCHMARK_20261014.md)。并未证明质量无损、稳定加速或并发容量提高。按用户选择继续[Lesson4批量执行](LESSON_04_BATCH_EXECUTION.md)，以下起点和Lesson1段落保留历史，不重复开发；未完成验证仍保留。

所有问题从 `/home/ubuntu/infer-engine/nanovllm_qwen35` 的真实适配出发。已有单卡纯文本 eager、Full KV 与 GDN 状态池、联合准入、不混批 Chunked Prefill。当前 Full Attention 历史仍通过 index_select 形成稠密张量，GDN 使用参考核心，多处逐请求执行。

本周目标：接通一个直接读取分页池的单 token Decode 后端，理解并减少逐请求执行；先正确后计时。不是重新学习 KV Cache，也不宣称手写 kernel 已达到工业性能。保留旧参考路径及显式后端选择；不升级现有环境，新增依赖另行确认。

## 六课安排（另留一次复盘/缓冲）

| 课 | 从当前代码切入 | 实作与交付 | 验收 |
|---|---|---|---|
| 1 分页算子契约 | pool_backend.py::forward 的 history_addresses/index_select；layer_compute.py 的 Full 路径 | 写明 Q、新 KV、池布局、块表、有效长度、scale、GQA、位置与写读顺序；盘点现有后端支持 | 手推跨页地址，区分逻辑位置/物理槽/层索引；解释为何无需读取无效尾槽 |
| 2 Decode 分页后端 | 同一 Full 层的新旧执行接口 | 在确认接口支持后接成熟分页后端，仅单 token Decode；Prefill 保留旧路径 | 对照稠密参考，覆盖非连续块、末块、空/短历史、异长双请求、换序、GQA |
| 3 正确性与成本 | GPU Runner 与真实模型验证入口 | 先层级 output，再模型 logits/生成及状态；比较临时张量与独立计时 | 不只看 argmax；记录 dtype/容差/有限值；不修改必要同步来制造加速 |
| 4 批量执行 | pool_backend 层内请求循环、gpu_engine.py::compute_logits | 先选一个投影或 LM Head 做批量化；Attention/GDN边界仍按请求隔离 | 解释 packed 输入不等于批量计算，检查矩阵形状改变的数值影响 |
| 5 kernel 教学支线 | 已固定的分页 Decode 契约 | 写最小 Triton/CUDA 分页读取与在线 softmax 原型（后端接通后才开展） | 解释最大值/归一化累积、归约与访存；正确但慢也如实报告，不要求超越成熟库 |
| 6 性能与调度复盘 | benchmark、schedule 的 Decode 提前返回 | 无 profiler 交错 A/B；独立短 trace；手推混批 token 预算和 sample_rows | 记录 TTFT/TPOT/吞吐/峰值显存与波动；不把分块当 PD 分离或自动混批 |

课5是条件性实作：主线未通过则只完成原理和接口推导，不同时调试两个新 kernel。课6的混批先设计，实施应独立于算子改动。首次准入仍预留完整 prompt KV 的现状必须注明。

## 每课 vLLM / InferLab 对照

先核实 `/home/ubuntu/vllm-main` 实际仓库根与 commit，再定位 V1 Scheduler、GPU ModelRunner、Attention backend、KV cache manager；重点看“谁生成 metadata、谁读取 block table、谁拥有状态”，不依赖未经核实的行号。

InferLab 只读 paged_kv_cache 与 decoder 的职责和所有权，讨论 Python tensor 生命周期对应什么 C++ RAII/缓冲所有权；本周不复制第二套引擎。

## 证据与范围

- GPU运行前检查空闲卡；没有可用卡时只做接口和CPU手推，不共享抢跑性能实验。
- 固定模型、输入/输出长度、batch、精度、预热、同步边界；编译与首次加载排除计时。
- 正确性与 profiling/benchmark 分开。小型断言/回归入口可以新增，暂存 trace 和张量快照不长期堆积；清理前留关键结果，已删除原始数据不能声称可逐次审计。
- FLA 保持隔离结论，不因原型性能好就直接替换生产参考核心。
- 每课3分钟复述 + 一个失败反例 + 一处学生可独立修改；助手代写不计为独立实现。

## 已学：Lesson1 分页 Decode 接口

先读 pool_backend.py 的 Full 分支：当前历史 gather 发生在层计算之前；新 KV 的生成、RoPE 与写回关系要对照 layer_compute.py 核实，不能直接删掉 index_select 就认为适配完成。

手推题：教学 block_size=4，A 的 block_table=[5,2]；本轮前已处理4个token，本轮输入第5个token。请回答：

1. 本轮 position、写入物理slot、写入后 context_len 各是什么？
2. Attention 有效读取的5个物理slot分别是什么？
3. 为什么位置编码不使用物理slot？为什么不读取第二块剩下3个槽？
4. 改为直接分页后，哪些输入/依赖不能随着稠密历史张量一起删掉？

4是教学块大小，不代表真实后端支持该配置。答题后盘点安装环境与后端签名，先选可兼容方案再写接线；当前没有新增 kernel 或安装依赖。

## 进度

- [x] 更新两周课程与入口。
- [x] Lesson1 接口概念经纠错通过：层切片、块表、长度、逻辑/物理位置、stream依赖。
- [x] Lesson2 助手完成分页Decode首版，学生已解释元数据上传与index_copy_写回；不等于独立实现。
- [x] Lesson3 已执行有限正确性检查和性能实验，证据及边界已记录。
- [ ] Lesson3 扩展验收：真实输入独立精度参考、长生成、换序/取消/失败路径；质量与性能结论未升级。
- [ ] Lesson4 批量执行（当前，从形状与源码开始，未改生产实现）。
- [ ] Lesson5–6，逐课执行。
