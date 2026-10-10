# 暂未实现的优化与恢复策略

2026-10-10最新进展：独立profiling已完成，并实施第一个受控优化——positions/slot_mapping每批上传一次、跨层复用；历史地址和全部安全检查保留。真实模型27步logits/状态精确对拍通过，单请求Decode H2D 50→12、双请求98→20。L64/N32交错A/B总吞吐中位数约+5.3%/+8.5%（B1/B2），TTFT无稳定改善。见[优化记录](../week4/lesson4/METADATA_REUSE_20261010.md)。第二阶段仅部分完成，剩余同步/历史地址和后续阶段尚未实施；下方2026-10-09条目是历史状态。

## 六阶段执行计划（2026-10-09）

保持单卡、纯文本 Qwen3.5-4B 参考路线，每阶段单独保留正确性与性能证据；不一次性重构全部热路径。

1. **性能基线与瓶颈证据**：先无 profiler 测 TTFT、TPOT、吞吐、显存，再独立采集 trace 区分 CPU 调度、host gap、H2D、GPU 算子。当前：8组GPU基线完成，关键数据保留在本文下方；独立profiling尚未开始。
2. **批次元数据与同步开销**：每批一次构造/上传 positions、slot_mapping、块表和状态槽；复用输入缓冲。分离诊断检查，只有证明生命周期安全后才减少同步，不删除必要的完成屏障。
3. **Full Attention 分页执行**：由历史 KV gather 成稠密张量改为直接读取分页池；覆盖块边界、非连续物理块、多请求和有效长度。
4. **GDN/conv 优化与批处理**：参考 torch chunk 保留为对照，接高效算子，减少逐层逐请求 Python 循环；验证状态写回、重用与混合长度。新增依赖单独确认。
5. **批量 LM Head 与采样**：合并逐请求投影；明确 BF16 GEMM 数值路径差异，验证 logits 误差及 token 一致性，不以放宽阈值掩盖错误。
6. **CUDA Graph 与调度升级**：先稳定地址和可捕获执行路径，再分步引入 Graph、Chunked Prefill/混合批次；分别测收益，不能把一个阶段的全部功能绑定成一次改动。

每阶段验收：资源与正确性回归、同一负载的基线对照、原始数据、波动和适用范围。以上为优化路线，不代表六项均已实现。

### 第一阶段测量协议

- 脚本：`benchmark_gpu_baseline.py`，必须从适配代码根目录运行。固定 max_model_len=512、max_num_seqs=2、block_size=256；保留当前安全校验和同步。
- 矩阵：输入 token 长度 19/64/128/256 × batch 1/2，生成32 token；每组2次完整预热、5次重复。先用一个小用例冒烟，再跑矩阵。
- 使用精确长度合成 token 序列、greedy、ignore_eos=True；这是闭环定长吞吐基线，不是自然 EOS/回答质量测试，也不是在线服务 SLA。
- TTFT：入队前开始至首 token 完成采样和结算，包含准入、清零、输入准备、执行及采样；不含加载、分词、网络和反分词。
- TPOT：每请求 `(末 token 时间-首 token 时间)/(生成数-1)`；原始 JSON 同时保留每次 token 间隔 ITL。报告 p95 是重复实验均值的描述统计，5次不能作为可靠尾延迟结论。
- 吞吐：主指标为全部生成 token / 批次总耗时；另列 Decode token/s，JSON另存输入+输出 token/s，禁止混称。
- 显存：每轮同步并 reset_peak_memory_stats，记录 resident allocated、peak allocated、peak reserved；包含权重/池/工作空间，不等同于逻辑缓存。不在计时内 empty_cache，不把 reserved 解释为泄漏。
- 关闭 capture_logits，不加载第二份 HF 模型，不进行 clone 快照/对拍/profiler。模型加载与所有预热排除计时。测量后独立 profiling，不混用两类时间。
- 记录环境、代码SHA256、Git状态、GPU编号/UUID和测试前后负载；执行前确认空闲卡，若有共享负载则降级为探索结果。现有脚本不会替用户抢占或自动挑选 GPU。
- JSON 每组落盘并保留失败状态；完成后自动生成同名 Markdown 报告；拒绝覆盖已有报告。

服务器运行示例（GPU编号必须先核实）：

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
nvidia-smi
CUDA_VISIBLE_DEVICES=<空闲卡编号> OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python benchmark_gpu_baseline.py --lengths 19 --batches 1 --new-tokens 4 --warmup 1 --repeats 3 --output baseline_smoke.json
CUDA_VISIBLE_DEVICES=<空闲卡编号> OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python benchmark_gpu_baseline.py --output performance_baseline.json
```

### 2026-10-09 基线关键数据（长期保留）

真实Qwen3.5-4B单卡eager；物理GPU2 NVIDIA L40，UUID `GPU-ff571342-a6b2-83bc-db0b-572401760627`。PyTorch 2.5.1+cu124、Transformers 5.8.0、驱动570.211.01，TF32关闭、CPU线程2；GDN为参考torch实现。固定容量512/2/256（max_model_len/max_num_seqs/block_size），每请求生成32 token，每组预热2次、测量5次，greedy且ignore_eos=True；测量内不含HF对拍、快照或profiler。

表中时间和吞吐为5次中位数；方括号为吞吐最小/最大值，显存为各组最大值。TTFT包含入队至首token结算，不含模型加载/分词/网络；TPOT=(末token时间-首token时间)/31；输出吞吐为整批生成token数/整批耗时。

| 输入长度 | Batch | TTFT ms | TPOT ms | 输出token/s [min,max] | Decode token/s | 峰值allocated GiB | 峰值reserved GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 1 | 130.18 | 42.63 | 21.88 [20.12,23.59] | 23.46 | 8.041 | 8.064 |
| 19 | 2 | 277.80 | 84.98 | 21.89 [21.70,22.76] | 23.54 | 8.041 | 8.064 |
| 64 | 1 | 138.66 | 41.77 | 22.58 [22.10,24.22] | 23.94 | 8.045 | 8.066 |
| 64 | 2 | 276.44 | 84.08 | 22.28 [21.83,23.01] | 23.79 | 8.045 | 8.068 |
| 128 | 1 | 128.10 | 38.20 | 24.37 [21.26,24.55] | 26.18 | 8.057 | 8.088 |
| 128 | 2 | 292.59 | 80.03 | 23.39 [21.27,24.15] | 24.99 | 8.058 | 8.088 |
| 256 | 1 | 158.03 | 43.98 | 20.97 [20.73,21.40] | 22.74 | 8.084 | 8.107 |
| 256 | 2 | 313.97 | 85.68 | 21.58 [20.48,23.37] | 23.34 | 8.084 | 8.107 |

- 常驻allocated约8.0166 GiB。显存包含权重、预分配池和临时张量，不是KV缓存大小。batch=1仍预分配2请求池。
- 常驻外部进程约243 MiB，整卡测前/测后约295 MiB、0%利用率：这是低负载共享卡数据，不是独占卡。1秒采样整卡最高9112 MiB，利用率0–52%、SM时钟2490 MHz、温度38–47°C；采样不能覆盖瞬时峰值。
- batch翻倍、TPOT近翻倍而总吞吐基本不变，符合当前逐请求执行结构，但尚未用profiler完成归因。不同长度结果有波动，不能据L=128较快推断长度越长越快；后续做交错A/B重复。
- 8组均完成，每组5次输出token序列一致；121项回归通过（含3项指标测试）。这不是本轮重新执行HF对拍，也不代表线上SLA或尾延迟。
- 基线脚本SHA256：`e69640c26e76960df82314fbacfba772c5bc122f98c4529c854d5c52d7401f30`。
- 按用户要求仅保留此摘要；本次正式/冒烟JSON、独立Markdown报告、CSV及日志已清理。原始逐次记录不再保留，无法重新统计本次分位数或核查逐token数据；可用保留的脚本复测，不能恢复完全相同的原始样本。下一步为独立profiler定位，尚未实施优化。

2026-10-08记录。全部为未完成项，不写成简历中的已实现收益；先真实正确性，后性能优化。

| 优先阶段 | 待办 | 前置条件与验收 |
| --- | --- | --- |
| 正确性基础 | GPU同步/事件、失败隔离、slot generation句柄 | 证明释放前无在途访问，旧句柄不能误写复用slot；不能套CPU同步声明 |
| 正确性基础 | 严格权重映射、dtype与布局验收 | 见BACKEND_AUDIT；不能靠同shape认为同语义 |
| 后续恢复 | 只重试采样，复用成功forward的logits | 保留有效logits及执行凭证，区分采样是否改变RNG/约束状态，防重复结算；不能重跑已更新状态的forward；评估logits保留显存 |
| 后续恢复 | conv/recurrent快照或从干净状态重算 | 快照需全层同一token边界并覆盖KV/计数；完整前缀重算验证等价；当前失败只回收，不恢复 |
| CPU管理性能 | 联合准入全块快照改预留日志/局部undo | 所有失败注入回归通过，再测调度开销；当前O(总块数)快照优先正确 |
| 调度扩展 | Chunked Prefill、混合批次、公平性、抢占 | 逐请求边界、预算、状态续算一致；不能只增加一个is_prefill分支 |
| 缓存扩展 | Hybrid Prefix Cache | 同一前缀边界的Full KV和conv/recurrent快照都可用；所有权、引用计数与淘汰共同验证 |
| GPU性能 | GDN/conv优化算子、融合、CUDA Graph、compile | 固定真实后端后对拍；测host gap/算子瓶颈/状态地址稳定性，再按证据启用；安装依赖另行确认 |
| 存储性能 | 输入缓冲复用、异步H2D、状态gather/scatter优化 | 生命周期与流依赖正确；高级索引产生拷贝时必须明确写回 |
| 工程完善 | completed/known_ids回收策略、故障隔离、有限性校验开销 | 保留输出交付与唯一ID保证，避免无界CPU增长；调试检查与热路径分开 |
| 扩展非首版 | KV压缩、TP、多模态、量化、PD分离、MTP | 单卡eager纯文本正确性验收后另立实验；当前不承诺两个月全实现 |

每项完成至少留：代码、失败/边界测试、可复现命令；性能项另留基线、负载、波动和质量代价。真实模型尚未接通前不报告吞吐收益。

2026-10-08更新：显式Qwen35LLM已接GPU池和保守同步生命周期，原始torch chunk Prefill已复用对齐，27步真实4B精确对拍通过。上表GPU同步基础部分已有首版，事件级异步与slot generation仍未实现。新增待优化：逐请求LM Head改回批量GEMM（已观察BF16数值路径差异）；分页历史gather改直接分页Attention；参考torch chunk改优化GDN。不得以首版成功代替性能测量或全场景正确性。
