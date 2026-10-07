# Week 4：nano-vLLM 框架与 Qwen3.5 适配学习计划

> **For agentic workers:** 按用户已选逐课、当前任务执行方式使用 superpowers:executing-plans；本文件是课程计划，不授权一次性代写全部引擎。每课实现前细化接口、失败测试和小范围补丁，再执行。

**Goal:** 学生能够独立追踪、解释并修改 nano-vLLM 核心链路，识别 Qwen3.5 混合状态适配点，并初步映射 InferLab 模块。

**Architecture:** nano-vLLM 为主读本；正式 vLLM V1 是逐课对照对象；Week 3 Transformers 路径作为参考后端和排错工具。先掌握控制与状态，再逐步替换模型执行，不把桥接运行冒充原生适配。

**Tech Stack:** 现有 nanovllm 环境、Qwen3.5-4B、单张 L40；先 eager、batch=1、纯文本。InferLab 只读 C++ 接口，不开启第二套引擎实现。

**Spec:** 本页“课程契约”与用户本轮要求；延续[原设计](../archive/development/superpowers/specs/2026-09-25-qwen35-text-baseline-design.md)，但按用户明确选择解除 Week 3 完整数值验收对框架学习的阻塞。

## 课程契约与进度

- [x] 建立课程、真实源码入口、正式 vLLM 对照和结课标准。
- [x] Lesson 1 概念验收：请求生命周期（见[验收记录](LESSON_01_ASSESSMENT.md)；独立源码 trace 尚未验收）。
- [ ] Lesson 2 调度与分页缓存。
- [x] Lesson 3 Runner概念验收经纠错通过；独立代码验收仍待完成。
- [ ] Lesson 4 Qwen3.5 模型与状态契约（当前课程，见[Lesson04](LESSON_04.md)；配置/层映射已落地，原生执行未实现）。
- [ ] Lesson 5 请求重排、释放和最小接入。
- [ ] Lesson 6 InferLab 映射、框架复述与独立改动。
- [ ] 完成[结课验收](ASSESSMENT.md)。

“掌握”以独立操作和解释验收，不以文件数量、看完源码或助手写完代码判定，不保证按日历自动获得能力。
建议6次主课加1次复盘；每课约2–3小时，可按学习速度调整。以用户答题与实作推进，不重复已掌握的 Attention 基础。

## Global Constraints

- 主目录 /home/ubuntu/infer-engine；参考 nano-vLLM 位于 /home/ubuntu/t1/nano-vllm-main/nano-vllm-main；InferLab 位于 /home/ubuntu/infra-main。
- 参考目录已有用户注释，当前只读，不直接改写；开始代码适配前固定来源和导入路径，在主目录内建立明确实现边界，避免导入环境里另一份 nanovllm。
- 正式 vLLM 以用户上传的 /home/ubuntu/vllm-main/vllm-main 源码为当前对照；其提交版本未确认，不等同于此前 v0.18.2 文档基线。源码入口见对照卡，不声称已运行正式 vLLM。
- 不安装/升级 vLLM、torch 或快算子，不在 nanovllm 环境混装正式 vLLM；如后续需要实跑，另行确认独立环境方案。
- Week 3 数值状态仍是 diagnostic；不能写“已通过”。框架学习可继续，原生适配正确性与性能结论仍需要对应证据。
- 首个 Qwen3.5 路径禁用未经验证的 prefix reuse、chunked prefill、抢占恢复与 CUDA Graph；现有代码不一定有这些开关，须实现显式拒绝/限制并测试，不能只在文档说关闭。
- 单卡 TP=1 不等于当前 runner 无分布式初始化；需要理解现有 NCCL 与端口行为，不未经检查启动多个 runner。
- 不把“短期/长期记忆”比喻当数学定义。conv state 是卷积所需的有限窗口状态，recurrent state 是递推状态，不是所有历史 token 的无损存档。

## 每一课都必须留下四项证据

2026-10-01新增执行要求：按[面试耦合计划](../INTERVIEW_PLAN.md)每课补一段3分钟独立复述和一道反例追问。Lesson3重点考packed输入、token/state两种slot及执行同步边界；Lesson4考模型接入和状态契约；Lesson5考错误恢复与所有权；Lesson6考上游/个人贡献及InferLab C++映射。面经只用于选学习问题，技术答案以实际源码、测试与官方技术资料验证。

1. nano-vLLM：真实文件、函数、输入输出与一次状态变化。
2. vLLM V1：本地对应职责入口、至少一个机制差别、来源路径；文档链接需注明其版本与本地源码不一定相同。
3. Qwen3.5：哪些假设不能沿用、此次改动属于控制层还是模型/算子层。
4. 验证：手推结果、运行证据、失败用例，以及未验证范围。

使用[对照表](FRAMEWORK_COMPARISON.md)，不只在周末笼统说“vLLM 更复杂”。

## Lesson 1：一个请求从提交到结束

**Read:** nanovllm/llm.py、engine/llm_engine.py、engine/sequence.py、engine/scheduler.py 的 add/postprocess。
**Files（学生后续创建）:** week4/notes/request_lifecycle.md。

- [ ] 跟踪 generate → add_request → Sequence → Scheduler.add → step → schedule → ModelRunner.run → postprocess。
- [ ] 区分 token_ids 中已选出的 token 与 num_cached_tokens；画出 num_scheduled_tokens 的一次归零。
- [ ] 给 prompt=[11,22,33]、max_tokens=2、无EOS、完整Prefill的情形画两轮 trace；第一轮后 num_tokens=4、num_cached_tokens=3，第二轮停止前分别为5、4；注意 deallocate 后缓存计数重置。
- [ ] 对照 vLLM V1 API/Engine Core/Worker 的职责分离，不把 nano-vLLM 单进程调用链照搬为生产进程架构。

**验收:** 学生指向源码解释“生成第二个 token 为什么不是第二次 Prefill”；指出采样者、追加者、释放者。
从[Lesson 01](LESSON_01.md)开始，不需要启动 GPU。

## Lesson 2：调度器和 BlockManager 各管什么

**当前课程：** [Lesson 02](LESSON_02.md)。从本课起，按[Qwen3.5 适配主线](QWEN35_ADAPTATION.md)把读源码、改 nano-vLLM、测试和正式 vLLM 对照绑定，不另开脱离项目的模型理论线。

**Read:** engine/scheduler.py、engine/block_manager.py、Sequence.block_table。
**Files（计划）:** week4/notes/scheduler_trace.md；实现练习时新增独立 CPU 测试文件，不改源目录。

- [ ] 手推 waiting/running、token budget、can_allocate、allocate、may_append、deallocate 的职责。
- [ ] 用 toy block_size=4、block_table=[5,2]，算逻辑位置5对应物理slot=2×4+1=9；toy数值不是该Config合法生产配置（当前要求256倍数）。
- [ ] 构造小预算和两个请求，指出当前实现只允许第一条已选请求跨预算分块，且 Prefill 有任务时该轮提前返回。
- [ ] 分别验证完整块前缀复用、引用计数归零、块耗尽；先用失败测试固定行为，再改策略。
- [ ] 对照正式 V1 的 token-budget 调度及 chunked prefill 策略；解释该本地版本已有分块而不是“完全不支持”。

**验收:** 解释连续批处理、分页、前缀复用不是同一个概念；解释只命中 Full Attention KV 不足以恢复 Qwen3.5 混合历史。

## Lesson 3：ModelRunner 如何把请求变成张量

**Read:** engine/model_runner.py 的 prepare_prefill/prepare_decode/run_model/run；utils/context.py、layers/attention.py、layers/sampler.py、utils/loader.py。
**Files（学生后续创建）:** week4/notes/runner_shapes.md。

- [ ] 标出 packed input_ids、positions、cu_seqlens、slot_mapping、context_lens、block_tables，不假定都为[B,T]。
- [ ] 跟踪 Prefill 的可变长度边界和 Decode 每请求一个输入；说明 token slot 与未来 recurrent slot 不是一回事。
- [ ] 查清 sampler 当前是带温度随机采样，不是 Week 3 argmax；不能把 temperature=0 随便传入当前除法路径。
- [ ] 区分 forward 与 compute_logits；先看 eager，再解释图捕获为什么要求地址与执行形状受控。
- [ ] 对照 vLLM V1 的 Worker/ModelRunner 职责。跨进程和性能优化只讲边界，本课不实现多卡。

**验收:** 给定两个长度不同的请求，能标出每个 token、每个输出及其状态归属；指出现有 run 中异常时 reset_context 的覆盖风险，设计测试后再修。

## Lesson 4：Qwen3.5 适配不是换模型名

2026-10-07起使用[Lesson04](LESSON_04.md)与 `/home/ubuntu/infer-engine/nanovllm_qwen35` 适配副本。外部参考目录只读；来源指纹、运行命令、实现边界见[副本说明](../nanovllm_qwen35/ADAPTATION.md)。先CPU配置契约，再状态接口和真实模型，不预先宣告原生适配完成。

**Read:** config.py、models/qwen3.py、utils/loader.py、ModelRunner.__init__/allocate_kv_cache；对照 Week 3 配置与正式 vLLM Qwen3.5 源码文档。
**Files：** week4/QWEN35_ADAPTATION.md 已建立逐课适配主线；本课细化模型与状态接口，再进行相应实现。

- [ ] 列出 text_config、模型类、权重名称/packed投影、RoPE/gate、EOS来源、采样、KV层数、linear状态的差异。
- [ ] 固定 Full Attention 原始索引 [3,7,11,15,19,23,27,31] 与紧凑KV索引0..7的映射；不能把32层都分普通KV。
- [ ] 设计 request_id → state_slot，以及 layer_id → conv/recurrent buffer；先描述所有权、初始化值和释放时机，再写类。
- [ ] 对照正式 vLLM 的混合缓存分组与模型接口，明确哪些只作参考，哪些实际实现。
- [ ] 区分M1桥接与M2原生接入：M1若仍调用完整Transformers forward，必须标注“引擎控制链桥接”；M2必须由适配模型消费引擎提供的状态与布局，并逐层检查加载与输出，不能只改import。

**验收:** 能指出至少6处真实适配点，并解释不支持功能如何被拒绝。设计确认后才修改代码；不承诺本周完整原生适配。

## Lesson 5：状态随请求走，不随 batch 下标走

**Files（计划）:** week4/notes/state_ownership.md、后续选定的state manager及对应测试；具体接口在Lesson4确定。

- [ ] 手推 A→slot2、B→slot5，batch [A,B]→[B,A] 后 gather顺序变[5,2]，scatter仍写回各自slot；不要求物理搬移所有KV。
- [ ] 用不同哨兵值验证 conv与recurrent状态、KV块表、长度、位置、采样参数和输出request_id均不串用。
- [ ] 验证A完成/取消/异常后释放其资源，B不受影响；C复用A的slot时不能读取A残留的有效状态；拒绝陈旧句柄与重复释放。
- [ ] 同一不可变前缀可在有正确引用计数/写入隔离的机制下共享；未建立这种机制时不可自行共享可变状态。
- [ ] 完成单卡单请求Qwen3.5最小控制链接入；保存真实短生成和状态轨迹，明确M1/M2边界。

**验收:** 学生亲自改一处状态映射、观察测试失败，再恢复并解释；完成两请求重排的CPU契约测试。真正GPU连续批处理、抢占与Graph延至后续，不把模拟测试称作并发吞吐验证。

## Lesson 6：InferLab 框架对照与结课

按[InferLab映射](INFERLAB_MAP.md)只读请求、调度、分页与decoder相关模块；先区分机制实验、真实模型执行和结果编排。
学生独立提交一页框架图与一项小修改，按[验收表](ASSESSMENT.md)评定。通过“框架掌握”不自动等于“Qwen3.5原生数值验收通过”。

## Review Focus

- 计数：采样append与forward处理混淆 → Lesson1两轮trace验收。
- 地址：token物理slot与request状态slot混淆 → Lesson2映射题与Lesson5重排测试。
- 复用：只命中KV却缺linear历史 → Lesson2/4明确拒绝不完整前缀复用。
- 生命周期：异常、重复释放、slot再用 → Lesson5故障用例。
- 来源：把桥接/上游已有功能写成原生新增 → Lesson4里程碑记录与结课来源说明。

## 现在的边界

当前进入Lesson4，已建立适配副本及CPU配置契约测试。未执行模型、不安装vLLM、不改外部nano-vLLM/InferLab源目录。实现放在主目录内固定来源的适配副本，逐课小改动、测试、讲解，不一次性代写完整引擎。
