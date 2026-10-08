# Lesson4：Qwen3.5模型与状态契约

日期：2026-10-07。主线：在真实nano-vLLM副本上逐步适配Qwen3.5-4B。不是换模型名，也不是一次性写完整引擎。

## 0. 前置验收记录

Runner概念验收经纠错通过：packed输入、Q/K累计边界、物理token slot、请求state slot、logits/采样/追加职责、异常清理与历史回滚区别。仍须巩固边界算术与独立实现；不把口头通过当成GPU模型通过。

学生已写build_layer_maps及正常结果断言；未知类型测试由助手补充。本课把它接到真实config.json，当前进度见[后端核查](../nanovllm_qwen35/BACKEND_AUDIT.md)。历史CPU测试汇总已归档，Week3数值容差仍未批准。

## 1. 文件与当前范围

- 适配代码：`/home/ubuntu/infer-engine/nanovllm_qwen35`。
- 只读来源：`/home/ubuntu/t1/nano-vllm-main/nano-vllm-main`。
- 真实配置：`/home/ubuntu/huggingface/Qwen3.5-4B/config.json`，不复制权重。
- [副本使用说明](../nanovllm_qwen35/ADAPTATION.md)；来源逐文件SHA256在SOURCE_MANIFEST.json。

配置检查和层映射已实现；后续已加入独立CPU同步StateManager，见[实现说明](../nanovllm_qwen35/STATE_MANAGER.md)。Qwen3.5模型、Runner接入和GPU状态执行仍未实现。保留上游源码和用户注释；来源无可确认commit，不虚构版本。来源未附LICENSE，分发前需补齐核实，不影响当前内部阅读来源记录。

## 2. 从外层配置进入文本配置

真实配置外层model_type=qwen3_5，含text_config和vision_config。纯文本适配读取text_config；存在vision_config不意味着本引擎已支持图片。

本次核查文本字段：

| 字段 | 值 | 用途 |
| --- | --- | --- |
| num_hidden_layers | 32 | 检查layer_types条数 |
| hidden_size | 2560 | 文本隐藏维度 |
| num_attention_heads / num_key_value_heads | 16 / 4 | Full Attention头结构 |
| head_dim | 256 | 使用配置值，不套hidden_size/heads |
| linear_num_key_heads / linear_num_value_heads | 16 / 32 | linear状态形状来源 |
| linear_key_head_dim / linear_value_head_dim | 128 / 128 | recurrent矩阵维度 |
| linear_conv_kernel_dim | 4 | conv窗口状态长度 |
| eos_token_id | 248044 | 模型停止语义，仍需与生成配置/分词器核对 |

不要因为存在full_attention_interval=4就无条件生成层列表；本阶段要求显式layer_types。长度不匹配或未知层类型直接报错，不静默猜测。

## 3. 两套层映射，不是序列边界

调用你写的build_layer_maps：

```python
full_layer_to_kv = {3: 0, 7: 1, 11: 2, 15: 3, 19: 4, 23: 5, 27: 6, 31: 7}
# linear原始层为剩余24层，各自映射到0..23
```

模型layer_id、KV紧凑层索引、linear紧凑层索引不能混用。cu_seqlens描述请求序列边界，与层映射无关。

```text
Full:   原始层号 → KV紧凑层索引 → 请求block_table → token slot
Linear: 原始层号 → linear紧凑层索引 → 请求state_slot → conv/recurrent
```

原始层6为linear索引5；请求A的slot=1时，概念上读state[5,1,...]。原始层7为KV索引1；请求B的KV地址取自B的块表，与B的linear slot无关。

## 4. 状态张量契约（设计，不是已分配）

Full KV沿用分页思想，只覆盖8层。每层K/V的具体轴顺序必须与attention kernel一致，不能仅凭元素总数相同就认为兼容。

拟定linear池布局：

```text
conv:      [24, capacity, 8192, 4]
recurrent: [24, capacity, 32, 128, 128]
```

其中8192 = 2×16×128 + 32×128。真实后端可能采用不同轴顺序；这是对齐Week3观察建立的候选接口，原生算子接入时再验证。capacity是请求slot容量，不是上下文长度。

重要：真实config声明model dtype=bfloat16、mamba_ssm_dtype=float32；Week3运行曾观察到BF16状态。配置字段不是运行张量dtype的证明。本次CLI保留两项声明，并把runtime_state_dtype标成UNVERIFIED。后续需检查参考张量、算子累加与持久状态的dtype，再决定字节预算与数值对拍。

不能先按所有32层分普通KV，再把linear状态当作附属变量。两类资源应分别预算、统一准入，且区分逻辑字节、唯一存储与allocator占用。

## 5. 所有权与生命周期

| 组件 | 拟定职责 |
| --- | --- |
| Config/契约解析 | 校验文本配置、层类型、结构尺寸、支持范围 |
| Runner/存储池构建 | 实际分配KV与linear张量，核对布局与dtype |
| BlockManager | 管理Full KV块号、引用与请求块表，不创建每轮GPU张量 |
| StateManager（CPU原型已实现） | 请求slot预留、初始化、所有权、归还；GPU在途安全与陈旧句柄保护待实现 |
| Scheduler | 两类资源都满足才准入；处理等待/失败/结束 |
| Runner | 按seqs顺序传递请求元数据，不每轮重置历史 |
| Linear层/算子 | 消费本层索引和请求slot，读取并更新状态 |

新请求首次使用或复用slot时初始化；同一请求续算保留状态。失败分配全退；执行失败可能已修改一部分层，清context不等于回滚。未实现快照时，不沿用脏历史重试；确认无在途GPU访问后回收，若重试则从干净状态重算完整前缀。

Prefix命中必须同时拥有相同边界的Full KV和conv/recurrent状态。首版先拒绝未验证复用；Chunked Prefill、抢占重算、Graph分别验收。本次仅建立契约，不声称已经启用上述功能。

## 6. 为什么先加拒绝检查，而不是直接替换模型名

副本`nanovllm/config.py::Config.__post_init__`在AutoConfig之前检查原始JSON；发现Qwen3.5即抛NotImplementedError，说明原生执行尚未实现。

这样不会让新配置误进现有Qwen3模型、普通32层KV分配或NCCL/GPU初始化。检查函数有CPU测试；上游Qwen3分支不主动拒绝，但本轮没有运行它做GPU回归。

接下来的适配顺序：

1. 配置契约与层映射（本次）。
2. 确定状态dtype/布局，设计最小StateManager及资源不变量。
3. 接入模型选择、权重名称/packed投影、RoPE/gate与Full/Linear执行。
4. 真实单请求对拍与失败清理；通过后才替换当前整体执行guard。
5. 多请求与高级缓存功能逐项开放。

M1完整Transformers forward桥接与M2原生模型必须分别记录，不把本次配置解析称为任一里程碑完成。

## 7. 对照正式vLLM与InferLab

正式源码根目录 `/home/ubuntu/vllm-main/vllm-main`：沿模型注册/模型实现、缓存规格和调度职责查找；当前不是把nano的batch级is_prefill直接复制进V1。文件版本与入口核查规则见[对照卡](FRAMEWORK_COMPARISON.md)。不要由“统一调度”推导“所有层使用同一种缓存”。

InferLab继续读PagedKVCacheManager的can_fit_request/add_request及释放接口：普通块资源可用，不等于混合状态slot可用。后续C++练习用RAII表达预留失败回滚，不在本课重写模型。

## 8. 实作与验收

从[适配副本说明](../nanovllm_qwen35/ADAPTATION.md)运行CPU测试与真实配置检查。测试覆盖学生示例、未知层、层数不匹配、缺字段、非法维度、头数不整除、不修改输入、dtype不擅自推断以及真实Config入口提前拒绝。

- [x] 学生完成层映射函数与正常映射断言。
- [ ] 学生独立运行并解释失败路径，新增一个自己的边界用例。
- [ ] 解释hidden_size/heads为什么不能替代配置head_dim。
- [ ] 解释config声明dtype与运行状态dtype不一致时如何核查。
- [ ] 明确原始层号、紧凑层索引、请求slot、token slot四种索引。
- [x] StateManager基础接口与所有权规则经课堂确认；CPU实现及12项新增测试已通过。
- [ ] 学生独立解释allocate的初始化与提交顺序，并运行自己的边界测试。
- [x] StateManager与真实BlockManager的CPU联合准入组件及失败回滚测试，见[下一部分](../nanovllm_qwen35/ADMISSION.md)。
- [x] 联合准入接到独立CPU HybridScheduler队列/预算、结束/失败回收，见[调度器接线](../nanovllm_qwen35/SCHEDULER.md)；18项新增测试通过。
- [x] CPU RunnerInputBuilder消费ScheduledBatch，构造token/KV地址/state_slots；13项新增测试通过，见[输入准备](../nanovllm_qwen35/RUNNER_INPUTS.md)。
- [ ] 真正的模型执行、LLMEngine接线与GPU安全生命周期接入。
- [x] CPU同步执行接口与合成后端验收，11项新增测试；见[执行接口](../nanovllm_qwen35/EXECUTION.md)。这不代表真实模型forward完成。
- [x] 本机真实权重文件头、参考Full/GDN接口和CPU缓存dtype探针核查；见[后端核查](../nanovllm_qwen35/BACKEND_AUDIT.md)。未运行完整模型，待办见[优化记录](../nanovllm_qwen35/OPTIMIZATION_BACKLOG.md)。
- [x] 纯文本权重映射与覆盖计划：426项匹配、共享head别名、显式排除vision/MTP；9项新增测试，全套85项通过。不是实际加载或forward通过。
- [x] 参数存储骨架与严格CPU加载器：真实426项文本参数全字节验证、head共享存储；新增7项测试，全套92项通过。见后端核查第7节；仍未实现forward。
- [x] 独立CPU单请求真实层计算：norm/Q-gate/partial RoPE、Full/GDN及Decoder层，小型参考对拍与缓存续算通过；新增12项，全套104项。见后端核查第8节；完整模型与分页池执行仍待接入。
- [x] CPU多请求池接线：Full层映射→块表/token slot→KV池，GDN层映射→state_slot→原conv/recurrent池写回；新增9项，全套113项。见后端核查第9节；真实4B端到端与GPU仍待验证。
- [x] 单卡L40真实4B短输入逐层诊断：原始参考有非零误差，GDN recurrent消融三步全层输出/logits为0差异；见后端核查第10节。M1标准尚未批准，M2只完成独立GPU参考路线，不是GPU引擎接线完成。

面试复述：用3分钟说明“普通KV分页为什么不足以适配混合模型”，必须指出真实文件、测试、未实现部分，不把助手实现当个人独立完成。
