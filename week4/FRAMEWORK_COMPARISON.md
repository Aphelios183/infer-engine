# nano-vLLM 与正式 vLLM：逐课对照卡

核查日期：2026-09-29。本地读本为 /home/ubuntu/t1/nano-vllm-main/nano-vllm-main，含用户注释；未取得可用Git提交，不能声称等同上游某个版本。
更新于2026-09-30：当前优先对照用户上传的 `/home/ubuntu/vllm-main/vllm-main` 源码。其Git版本未确认，不可标为v0.18.2；以下v0.18.2链接仅保留为历史文档读本。本轮没有安装或运行正式vLLM。

## 当前本地 V1 源码入口

以下路径相对于上述正式vLLM根目录，行号是本次阅读定位提示，后续以函数名为准：

- `vllm/v1/engine/llm_engine.py`：`add_request`（223）、`step`（306），输入与输出侧职责。
- `vllm/v1/engine/core.py`：`step`（630），schedule → execute_model(non_block=True) → grammar mask → future.result → update_from_output；`step_with_batch_queue` 是另一条执行路径。
- `vllm/v1/core/sched/scheduler.py`：`schedule`（562），每请求待计算量与共享预算；`update_from_output`（1967）结算；`_update_request_with_output`（2434）追加与停止；`_free_request`（2650）及后续函数处理释放。

Lesson1纠错：grammar bitmask是结构化输出约束，不是训练正则化。非阻塞提交提供CPU/GPU重叠机会，不证明实际重叠或多轮流水；`future.result()`仍可能等待。请求接入也不能全部归到EngineCore.step。资源释放可能因在途执行或connector延后，不能把nano的同步释放照搬过去。

Lesson2注意：源码的`num_new_tokens`还涉及output placeholders、token/input预算、模型长度和混合状态对齐等限制；“追上num_tokens_with_spec”是理解核心，不是完整生产公式，也不表示所有输入已经执行完成。

## 1. 请求入口与执行职责（Lesson 1/3）

本地：LLM继承LLMEngine，generate同步循环调用step；rank0创建ModelRunner，TP>1时另起进程。本地有多进程代码，不能概括成“nano-vLLM没有多进程”。
正式V1区分输入/API侧、Engine Core和GPU Worker职责。学习重点是工作由谁负责，不是强行把两边同名类一一对应。[架构来源](https://docs.vllm.ai/en/v0.18.2/design/arch_overview/)

待学生回答：tokenization、调度、GPU forward、输出处理各自在哪？为什么LLMEngine不等于模型？

## 2. 调度（Lesson 2）

本地Scheduler.schedule先尝试Prefill，有已选请求就返回；否则进入Decode。已有按token预算分块的代码，但同轮用一个is_prefill标记。
正式V1以每请求token预算统一表达调度；选定版本的chunked prefill文档描述先安排Decode，再利用剩余预算放Prefill，可同批混合。[V1概念](https://docs.vllm.ai/en/v0.18.2/usage/v1_guide/)、[调度与调优](https://docs.vllm.ai/en/v0.18.2/configuration/optimization/#chunked-prefill)

待学生回答：长Prefill到来时，两种策略对已有Decode的等待有什么可能影响？这是待测推断，不是本项目实测加速结果。

## 3. 缓存（Lesson 2/4/5）

本地BlockManager维护空闲块、引用计数、前缀hash；ModelRunner按num_hidden_layers分配普通KV。没有看到请求级conv/recurrent状态管理。
正式V1提供KVCacheManager、组协调与按类型管理的层次，用于不同缓存需求；prefix命中不能只考虑一个组。具体模型和版本的功能限制须看代码，不把设计文档描述当所有模型均已支持。[混合缓存设计](https://docs.vllm.ai/en/v0.18.2/design/hybrid_kv_cache_manager/)

待学生回答：仅有相同tokens的Full Attention块，为什么不能直接跳过Qwen3.5整个前缀？

## 4. 模型适配（Lesson 4）

本地runner硬编码Qwen3ForCausalLM；Config读取顶层max_position_embeddings；普通KV预算使用全部层数。对Qwen3.5均需重新核对。
正式版本有Qwen3.5专用模型入口，ConditionalGeneration包装中创建文本language_model；应追踪其模型接口与权重映射，而非复用Qwen3类名。[模型源码文档](https://docs.vllm.ai/en/v0.18.2/api/vllm/model_executor/models/qwen3_5/)

待学生回答：桥接完整Transformers forward和原生模型适配各自证明了什么？

## 5. 执行与采样（Lesson 3/5）

本地prepare_prefill采用packed token布局，prepare_decode每请求一个token；context传递slot_mapping与块表。sampler使用温度与随机数，不能等同Week3贪心。
每课去正式架构文档对应的ModelRunner/Worker入口，核对输入准备、模型执行、采样边界；不要复用旧V0教程的类名来推断V1。
本周只解释Graph/并行边界，不声称覆盖正式vLLM全部优化或部署能力。

## 固定的课堂记录模板

```text
课次：
nano-vLLM 文件/函数/来源指纹：
输入和输出：
一次状态变化：
正式 vLLM 版本/入口/链接：
相同职责：
至少一个机制差别：
Qwen3.5 需要改变的假设：
本次已测范围/未测范围：
```

路径不可访问时标“待核查”，不补造实现细节；升级读本时保留旧版本说明。
