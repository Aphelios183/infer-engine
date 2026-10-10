# Lesson4 子课程：从读懂结构到完成一次优化

2026-10-09。上级入口：[LESSON_04](../LESSON_04.md)。本目录是接下来的五课主线；上级文档保留此前开发历史，不再按历史测试清单逐条学习。

## 学习约定与当前边界

- 每课按“读指定函数 → 预测数据/状态变化 → 解释原因 → 必要时改一处并观察”进行。助手不预先替学生完成全部改动；口头理解和代码已运行分开记录。
- 不把阅读助手生成的测试文件当作先修要求。历史助手测试与对拍结果从工作目录移至归档，保留真实模型验证入口，改动后的必要正确性验证不会取消。
- 当前是真实Qwen3.5-4B、单卡、纯文本、eager；显式入口为 `nanovllm.qwen35.Qwen35LLM`。它复用nano的Sequence/BlockManager等基础设施，不等于原始 `nanovllm.LLM` 全部迁移。
- Full层虽然用分页KV池，但计算时仍gather历史形成稠密缓存；GDN用参考torch chunk/递推。每层逐请求执行，未支持调度层Chunked Prefill、抢占、Graph、TP或多模态。
- 保留[六阶段优化路线和关键基线](../../nanovllm_qwen35/OPTIMIZATION_BACKLOG.md)。本五课是教学顺序，不承诺同时实现六阶段全部功能。
- 只有学生完成对应讲解/操作后才标记通过。目前五课均未验收，第1课开始。

代码根目录：`/home/ubuntu/infer-engine/nanovllm_qwen35`。以下源码路径均相对此目录；读源码优先，暂不展开其他开发文档。

## 小课1：完整请求链路与所有权（当前课）

目标：不只背模块名，能指出谁拥有资源、谁更新张量、谁提交计数。

按顺序阅读：

1. `nanovllm/qwen35.py`：导出的入口是什么？为什么不直接用原始LLM？
2. `qwen35_adapter/gpu_engine.py::Qwen35Engine.__init__/add_request/step/generate`：初始化与每轮执行分开；add_request只登记，不等于完成资源分配或GPU计算。
3. `qwen35_adapter/scheduler.py::HybridScheduler.schedule` 与 `admission.py::AdmissionController.try_admit/reserve_decode`：队列、预算、联合准入；一轮只允许一个未结算batch。
4. `qwen35_adapter/runner_inputs.py::RunnerInputBuilder.prepare`：构造token、位置、累计边界、块表、写槽和state_slots，不执行模型。
5. `qwen35_adapter/gpu_engine.py::GPUExecutionRunner.run`：forward、选取last_hidden、LM Head、采样、同步、postprocess。
6. `qwen35_adapter/pool_backend.py::PooledCPUBackend.forward`：GPU子类复用这里的流程；类名含CPU不意味着当前GPU路径在CPU算。找到按层/按请求的两层循环。
7. `qwen35_adapter/scheduler.py::postprocess/_finish`、`admission.py::release`、`gpu_engine.py::GPUStateManager.release`：append、计数、终止和资源归还。

本课必须亲自找到的两种写回：

- Full：原始layer_id → compact KV层索引 → 请求块表/slot_mapping → K/V池的 `index_copy_`。
- GDN：原始layer_id → compact state层索引 → 请求state_slot → conv/recurrent池切片的 `copy_`。

练习：prompt=[11,22,33]，max_tokens=2，假设依次选出44、55，均非EOS。分别说明add_request之后、首次schedule之后、prefill forward之后但postprocess之前、追加44之后、处理44之后、追加55并结束之后：请求token数、已处理token数、资源所有权发生什么变化。不要把“张量已经更新”和“调度计数已经结算”混为一谈。

验收：学生能定位各函数，并解释为什么45等新选出的token尚未进入缓存；能解释forward已更新GDN后为什么不能直接重跑同一批次。再交换两个请求顺序，解释token slot与state_slot如何随请求一起变换。

第一道问题：`add_request()` 返回后，是否已经分配了KV块和state_slot？指出实际分配发生在哪条调用链。先回答这一题，不提前做全部练习。

## 小课2：用profiler解释基线

2026-10-10：独立profiling已完成，见[实测时间线解读](PROFILING_20261010.md)。已采集CPU/CUDA时间线，插桩前后输出一致；尚未实施优化，继续对照实测证据学习。

目标：把结构与耗时对应，不凭GPU利用率或直觉判定瓶颈。

读 `benchmark_gpu_baseline.py::run_case`、`GPUExecutionRunner.run`、`PooledCPUBackend.forward/_check_inputs`。已有单请求TPOT约38–44ms、双请求约80–86ms，总吞吐约21–24token/s；这是低负载共享卡基线，不能当独占环境结论。

操作安排：先提出一个瓶颈假设；再在独立profiling运行中标记schedule、prepare、forward、LM Head、采样/同步、postprocess。选少量prefill/decode步看CPU/CUDA时间线，检查重复H2D、逐请求循环、同步点及kernel之间空隙。profiling前核实空闲卡。

验收：拿一段trace指出证据，区分CPU提交时间、GPU kernel时间、同步等待和端到端耗时；不能把异步CPU区间直接当GPU计算时长。定位一个最值得改的目标，并给出可能推翻自己判断的证据。不将profiler耗时混入正常基线。

## 小课3：一个小优化的完整闭环

2026-10-10：已完成positions/slot_mapping轮内复用及真实GPU验证，见[本次优化闭环](METADATA_REUSE_20261010.md)。单请求Decode H2D从50降至12；无profiler交错A/B吞吐中位数提高约5.3%/8.5%（B1/B2），TTFT无稳定改善。实现验证完成不等于学生独立验收通过；继续解释张量生命周期与复用边界。

目标：独立解释一次改动，而不是一次性替换整个后端。

候选是批次元数据一次构造/上传、输入缓冲复用或减少重复校验；依据上一课证据选择，不预设一定有效。重点阅读 `RunnerInputBuilder.prepare`、`PooledCPUBackend._check_inputs/forward` 中CPU/GPU张量边界与生命周期。

操作安排：先写下不变量（请求顺序、位置、有效长度、slot所有权），修改一处；用保留的真实模型验证入口或从归档恢复所需最小测试检查，再以同负载交错A/B测量。禁止为提升数字删除必要同步/错误检查而不提供等效安全保证。

验收：说明减少了什么操作、数据什么时候可复用；展示正确性与耗时对照。如果无收益，保留结论或撤回改动，不挑最好一次作宣传。

## 小课4：接一个高效执行模块

根据前两课证据只选一条：

- Full Attention：从pool_backend的历史gather出发，学习直接读分页池的接口，明确block_table、context_lens、slot_mapping、head布局；再对照当前算子支持范围，接合适后端。
- GDN/conv：从 `layer_compute.py` 的状态递推和prefill分支出发，区分计算中间态与持久池；核对高效后端布局、dtype、初始状态和最终状态写回。

本课不要求从零手写所有kernel，不同时开展多模态、FP8/TP或MTP。安装依赖前单独确认。

验收：能解释后端输入输出与池映射；单请求、多请求、边界长度、状态复用不串请求；对照参考数值路径并测同条件收益。若BF16计算路径改变，先分析误差而不是只放宽阈值。

## 小课5：调度层Chunked Prefill

目标：一个请求分多轮处理，Full KV与GDN仍表达同一历史边界。

阅读 `HybridScheduler.schedule/postprocess`、`ScheduledBatch`、`RunnerInputBuilder.prepare`、`PooledCPUBackend.forward` 的past处理及GDN初始状态分支。当前只支持完整prefill和单token decode，不能只改token_budget就声称完成。

设计要求：记录每请求本轮输入范围和实际已处理量；中间chunk不采样、不append新token；只有最后一个prompt chunk产生首个生成token；后续chunk沿用原conv/recurrent，位置从历史边界继续。Runner采样行与调度结算协议必须一并调整，不能继续假定每个请求每轮都产出token。

先做单请求分块，再做多请求预算；prefill/decode混批作为单独扩展，不在基础分块尚未对齐时一起接入。部分prefill失败也可能已经改变状态，仍需保证安全回收。

验收：比较完整prefill与分块路径的输出及两类缓存；解释每轮预算、输入范围、是否采样、状态是否初始化。避免把GDN算子内部chunk与调度器分块混淆。

## 完成后进入InferLab

门槛是独立追踪请求、完成一次有证据的优化、解释并验证混合状态续算与回收。无需先补齐多模态、FP8、TP、MTP、KV压缩。届时以资源管理、执行后端和调度边界对照InferLab，而非复制一套名词。

历史清理：助手生成的 `nanovllm_qwen35/tests/` 与4份旧GPU对拍JSON归档到 `/home/ubuntu/infer-engine/archive/lesson4-structure-20261009/assistant-tests-and-results.tar` 后从活动目录移除；历史“121项通过”只代表当时运行记录，不再表示活动目录现有测试数量。保留源码、基线脚本、GPU对拍入口、用户早期课程与手写代码。

## 当前学习位置与下一步（2026-10-10）

小课1链路和小课2同步/搬运基础问答已完成；小课3完成助手实施的positions/slots优化及实测，学生已正确解释每请求历史地址与空历史。尚不等于学生独立实现全部路径。历史addresses轮内复用暂未实现，5/6次H2D仅为理论目标。

现在进入小课4的结构准备，先掌握：

1. 分页存储不等于直接分页计算：追踪block_table → addresses → index_select → 稠密历史 → attention，找到临时张量及复制。
2. 明确高效Attention接口：Q、新K/V、KV池布局、块表、有效长度、位置与因果边界；页内偏移和层索引不要混淆。
3. 理解缓存写入与Attention读取的先后依赖；批量请求边界、空历史及跨块读写不能改变语义。
4. 再选一个后端作小范围集成，先对拍再测性能，不同时引入GDN、Graph和混批。
5. 小课5再进入Chunked Prefill：中间chunk不采样，末chunk才选首token，KV和GDN状态必须对齐同一已处理边界。

当前题：`kpool.index_select(0, addresses)` 返回的是原池view还是新张量？由此能否说当前Full Attention已经直接在分页KV池上计算？

本次清理仅删除生成trace/summary/临时采样文件及Python字节码，保留课程关键指标、实验脚本、小型A/B与正确性JSON。无自动commit或stage。
