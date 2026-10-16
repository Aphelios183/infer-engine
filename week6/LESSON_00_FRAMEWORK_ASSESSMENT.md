# 综合验收：串起 Qwen3.5 推理引擎

日期：2026-10-16。状态：已建课，尚未验收；当前从第1关开始。

## 学习目标与方式

目标是短时间建立推理框架技术栈认识，不继续追求大量kernel优化。
本课采用同一组请求贯穿调度、Runner、KV/GDN状态、采样、结束与复用。
每次只回答一关；先作答，再纠错并解释原因。没有预填答案，不把助手实现记为学生掌握。
本课仅手推与源码阅读，不运行GPU，不改生产代码。通过后做源码职责定位和最小Graph实验。

## 统一约定

- 使用当前适配的行为，不假定等同于正式vLLM：同质Prefill/Decode批次、不混批、已有Decode优先、无抢占、无prefix复用。
- `max_num_seqs=2`，每轮总token预算4，`prefill_chunk_size=2`。
- A先入队：prompt为 `[11,12,13,14,15,16]`，max_tokens=3。
- B后入队：prompt为 `[21,22,23]`，max_tokens=2。
- EOS ID为99，ignore_eos=False。题中给定采样结果，不考概率计算。
- A每次应采样时依次得到71、72、73；B依次得到81、99。
- 教学页大小4，资源足够；首次准入预留完整prompt的KV块。
- 题目指定准入结果：A的block_table=[5,2]、state_slot=1；B的block_table=[7]、state_slot=3。
- 这些物理编号是题设，不要求真实分配器恰好分配这些编号。页4仅用于手算，不代表Flash后端支持。
- 位置从0开始。context_lens包含本轮将处理的token；processed/cache计数在成功结算后才推进。
- Full层只读取有效KV；GDN的conv与recurrent必须一起与进度对应。
- 讨论失败时需明确GPU是否已完成；不能假设异常自动回滚已修改的状态。

## 第1关：入队不等于执行

时刻T0：只完成两次add_request，尚未调用schedule。

填写下表，并说明此时请求是否拥有KV块或state slot（上面的分配结果尚未生效）：

| 请求 | token_ids | num_tokens | num_cached_tokens | waiting/running/finished |
|---|---|---|---|---|
| A | 待填写 | 待填写 | 待填写 | 待填写 |
| B | 待填写 | 待填写 | 待填写 | 待填写 |

追问：真正协调KV与state联合准入的是哪个模块？若state无空闲slot而KV已预留，应怎样处理？

## 第2关：从调度批次到Runner输入

只讨论第一轮schedule：写出批次请求顺序、is_prefill、input_counts、sample_rows。
根据题设块表填：

```python
input_ids = []
positions = []
cu_seqlens_q = []
cu_seqlens_k = []
context_lens = []
slot_mapping = []
state_slots = []
last_indices = []  # 当前实现只选择需要采样的请求
```

解释：sample_rows是批次请求行号，last_indices是packed hidden行号，不能混用。
追问：第一轮每个请求的GDN状态应初始化还是沿用？每一层为什么还需要独立层索引？

## 第3关：区分forward完成与postprocess完成

沿用第一轮。分别描述：

1. forward已成功、postprocess尚未执行：物理KV/GDN是否已改变？调度器cached计数是否已提交？
2. postprocess成功后：A/B的已处理长度、总token数、生成数量如何变化？
3. 本轮无采样时是否仍需postprocess？它推进什么？
4. 如果forward完成后采样失败，能否用相同输入直接重试forward？如何避免状态重复更新？

## 第4关：完整推进两个请求

从第一轮开始，逐轮手推直到A、B都结束。不要默认两者每轮都执行；遵守Decode优先及不混批。

| 轮次 | 请求顺序 | Prefill/Decode | 本轮各输入token | positions | sample_rows | 选出的token | 结算后有效处理长度 | 请求是否结束 |
|---|---|---|---|---|---|---|---|---|
| 1 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 2及以后 | 自行续表 | | | | | | | |

重点口述：

- prompt未完成的请求能否采样？
- B已经进入Decode时，A的剩余Prefill会怎样被调度？这是策略取舍，不是自动混批。
- 停止token是否一定进入缓存？区分选出、输入模型、更新状态三个事件。
- 完成后的输出token仍保留供解码，资源释放不等于输出列表清空。
- `_finish`保存的Completion.processed_tokens与资源释放后Sequence内部计数，必须区分观察时刻。

## 第5关：回收与失败反例

1. B完成后C复用了B的state_slot。B残留的conv/recurrent可直接给C使用吗？何时初始化？
2. KV页含有旧数据时，为何维护有效长度可能足够，而GDN不能只改长度？
3. GPU操作尚未完成时，为什么不能把相关块/slot交给C写入？
4. 准入失败与执行失败有何区别？前者应保持什么不变量，后者为什么不能只撤销计数？

## 第6关：扩展架构迁移题

本关是设计题，不代表这些能力已经接入项目。

- Graph：固定输入缓冲区的copy_与Python重新赋值有何区别？GPU动态slot读取与Python固定切片有何区别？padding如何避免污染GDN？
- TP：16个Q头、4个KV头、head_dim256做TP2，各卡Attention输出宽度多少？o_proj后为何求和，残差何时加？
- DP/EP：模型副本、权重分片和专家分布分别是什么？每token只选两个专家为何不等于只存两个专家？
- PD：源端处理完100个prompt token并选出A后，传哪些状态？目标端逻辑位置是什么？物理块号能否照搬？
- 投机：前缀长度100，接受A/B、拒绝C并选出D，恢复到处理完B后，写出已处理长度、下一输入和逻辑位置。
- 多模态：仅做未学内容盘点；后续补processor、视觉特征、位置和调度预算，不列入本次必答通过条件。

## 源码定位与通过标准

当前适配入口（相对项目主目录）：

- `nanovllm_qwen35/qwen35_adapter/gpu_engine.py`：add_request、step、GPUExecutionRunner.run。
- `nanovllm_qwen35/qwen35_adapter/scheduler.py`：schedule、postprocess、_finish。
- `nanovllm_qwen35/qwen35_adapter/admission.py`：try_admit、reserve_decode、release。
- `nanovllm_qwen35/qwen35_adapter/runner_inputs.py`：RunnerInputBuilder.prepare。
- `nanovllm_qwen35/qwen35_adapter/pool_backend.py`：forward与池访问。
- `nanovllm_qwen35/qwen35_adapter/layer_compute.py`：Full/GDN层计算。

通过标准：能独立填写生命周期表；纠错后解释计数与状态边界；指出一个串请求/重复更新反例；能把职责定位到源码。之后再核实本地vLLM版本与对应入口，不预填未经核实的行号。

进度分别记录，不混同：

- [x] 助手建立验收课程。
- [ ] 第1–3关：入队、Runner与结算边界。
- [ ] 第4–5关：完整生命周期、回收与失败。
- [ ] 第6关：扩展架构迁移。
- [ ] 学生独立复述与源码定位。
- [ ] 下一阶段：最小Graph实验（未开始）。
