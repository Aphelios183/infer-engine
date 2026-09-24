# Week 3：从单层 Attention 到真实模型生成

更新日期：2026-09-24。按你的要求进入第三周。本周目标：用已有环境与本地小模型，跑通并解释一条真实生成链路，不以重写完整推理引擎为目标。

当前仅完成计划和只读环境核查；下述脚本、生成结果和追踪报告都属于待完成产出，不能当作已验收。

## 1. 本周范围与起点

你已经学习了 MHA、KV Cache、缓存生命周期、GQA 显存估算，并完成第二周性能实验与结果讨论。第二周最终综合题的前三项没有提交完整答案，不将其标记为通过；第三周实际读取模型配置时顺带复习，不阻塞本周开始。

本周只做：单卡、一个请求开始、短提示词、少量新 token；先看正确性与调用链，再看简单时序。不做调度算法改造、分页缓存实现、多卡、量化、CUDA kernel 或 InferLab C++ 重构。

建议总投入 12–16 小时，分 7 天推进。遇到兼容性问题保留一天缓冲，优先完成“一个请求能解释清楚”，不要靠增加功能凑进度。

## 2. 已核查的真实环境

| 项目 | 当前观察 |
|---|---|
| 主仓库 | /home/ubuntu/infer-engine |
| 环境解释器 | /home/ubuntu/enter/envs/nanovllm/bin/python |
| 本地模型 | /home/ubuntu/huggingface/Qwen3-0.6B |
| 模型文件 | config.json、tokenizer 文件、model.safetensors 均存在；尚未加载验证权重 |
| 初始阅读源码 | /home/ubuntu/t1/nano-vllm-main/nano-vllm-main |
| 另一份源码 | /home/ubuntu/t1/nano-vllm-p0；暂不混用，差异尚未核查 |
| Transformers | 环境中为 5.8.0；模型配置记录的保存版本为 4.51.0，版本不同不等于必然不兼容 |
| nanovllm 导入 | 从主仓库执行 find_spec('nanovllm') 返回 None，源码存在不等于此环境已能导入 |

本周先复用本地权重，不下载新模型。先用本次任务的 PYTHONPATH 或明确源码入口解决导入定位，不直接 pip install、升级或降级已有环境。确需依赖变更时，先给出错误证据与最小变更方案再确认。

该 nano-vLLM 分支的 SamplingParams 明确禁止 temperature=0；sampler.py 使用带温度随机采样，并有 torch.compile。不能照搬“temperature=0 就是贪心”的其他库用法。enforce_eager=True 关闭该分支的 CUDA Graph 路径，不等于禁用了所有编译。

## 3. 这周需要串起来的链路

```text
用户文本
  → 一次 chat template 格式化
  → tokenizer 编码成 token IDs
  → 请求创建与 Prefill
  → embedding → 多层 Transformer → 最终归一化 / LM head
  → 最后有效位置的 logits
  → 选出下一个 token
  → 把选出的 token 作为下一步输入，使用历史 KV
  → 重复 Decode，直到 EOS 或新 token 数达到上限
  → 输出 token IDs 解码为文本，请求结束并回收逻辑缓存资源
```

注意：Prefill 的最后位置 logits 已经用于选出第一个输出 token；只有把这个输出 token 再送入模型处理时，它自己的 K/V 才写入缓存。停止时最后采样出的 token 未必已经再经过一次 forward。

## 4. 七天安排

| 天 | 核心问题 | 动手任务 | 待产出与验收 |
|---|---|---|---|
| 1 | 模型配置和 token 到底是什么？ | 确认源码导入路径；读取本地 config；CPU 加载 tokenizer，观察模板、IDs、EOS 和解码 | ENVIRONMENT.md、inspect_model.py；记录真实路径与版本，解释文本与 token 数不相等 |
| 2 | 一次真实 forward 如何产生下一个 token？ | 用 Transformers 建立单请求参考路径；先跑短输出，再手动取 logits 的 argmax，记录第一步 | generate_reference.py；解释 logits 的位置维和词表维，保留输入 IDs 与输出 IDs |
| 3 | 真实模型缓存是否与完整重算等价？ | 同模型、同 dtype、同一固定 token 前缀，比较 use_cache=False 全前缀与缓存增量的最后位置 logits | verify_generation.py；逐步记录误差、top-1 和有效长度，不只比较最终文本 |
| 4 | nano-vLLM 中一次请求走过哪些函数？ | 先以原生支持的正温度跑通相同提示词；沿入口、调度、runner、模型、采样、结束追踪 | trace_request.py、REQUEST_TRACE.md；能定位 Prefill/Decode 转换和停止条件 |
| 5 | 教学实现与真实模型差在哪？ | 读 Qwen3 的 GQA、RoPE、RMSNorm、MLP、residual 和投影形状；读取真实缓存布局 | MODEL_NOTES.md；算出本地模型的 KV 字节，解释配置中的 head_dim |
| 6 | 能否独立复现和解释？ | 三条短提示词重复验证；记录设置、源码版本/指纹、错误与解决过程 | GENERATION_REPORT.md；证明运行可复现，明确未测/未对齐部分 |
| 7 | 综合验收与缓冲 | 修补薄弱点；独立跟踪一个新提示词；为下一周提出请求调度问题 | 完成验收清单，决定是否进入 Week 4 |

文件名是计划中的交付物；当前只创建本计划，不预先生成空脚本或虚构运行记录。

## 5. 第一天具体怎么学

先读 /home/ubuntu/huggingface/Qwen3-0.6B/config.json，不加载模型到 GPU。已看到的配置是：

```text
layers = 28
hidden_size = 1024
query_heads = 16
kv_heads = 8
head_dim = 128
vocab_size = 151936
配置 dtype = bfloat16
```

第一项重要区别：这里 1024/16=64，但模型显式配置 head_dim=128。不要强行套用第一周教学 MHA 的 C/H；当前 qwen3.py 优先采用配置中的 head_dim，Q 投影总宽度是 16×128，再经输出投影回到 hidden_size。

当天练习：

1. 本地模型每个 Q 头如何共享 K/V？每组有几个 Q 头？
2. BF16 每元素 2 字节，单请求缓存 4096 个位置时，全 28 层 KV 需要多少 MiB？按层累加，不用旧题的 32 层配置。
3. 同一条短文本格式化前后分别有哪些 token？模型真正接收的是哪一串 IDs？
4. 输入 L 个 token 的参考模型 logits 形状是什么？为什么取最后有效位置来选下一个 token？

tokenizer 实验固定一条短提示词，比如“用一句话解释 KV Cache”。明确记录 chat template 与 thinking 设置；如果本地模板支持关闭 thinking，首轮采用关闭并记录，避免把思考内容长度误当模型卡住。模板只应用一次，并检查后续编码是否重复添加 special tokens。

## 6. 正确性对拍必须怎么设计

先在同一个 Transformers 模型内对比全前缀重算和缓存增量。起步 batch=1、无 padding，先避开复杂 mask 和变长批处理。

- 固定模型权重、dtype、模板和一份输入 token IDs，eval + inference_mode；缓存每个独立用例重置。
- 先比较 Prefill，再把同一个下一 token 分别送入两条路径，比较对应位置 logits。固定前缀（teacher forcing），防止一次选 token 不同后，后续输入都不相同。
- 记录最大绝对误差、适当的相对误差和 top-1 结果。容差要结合实际 dtype 和算子路径声明，不盲用第二周 FP32 阈值，也不为了通过而任意放宽。
- argmax 不同但 logits 接近时，检查前两名分数间隔；只看生成文本相同也不能证明张量正确。
- 按当前 Transformers 版本确认缓存对象、position/mask 接口，不假定旧版 tuple API 一定适用。RoPE 的位置必须随着历史长度正确增长。

跨 Transformers 与 nano-vLLM 的 logits 对拍是进阶目标，不是首日门槛。若做对照，需在采样前捕获 logits，并确保相同输入 IDs、位置、权重和精度；不能用两个独立随机采样得到的文本是否相同判断正确性。固定随机种子也不保证不同实现采样出相同 token。

本周无需为了贪心采样修改 nano-vLLM 的 sampler。参考手动循环可用 argmax；nano-vLLM 先保持其原生正温度采样，清楚说明两者用途不同。

## 7. 源码阅读顺序

以下文件相对于 /home/ubuntu/t1/nano-vllm-main/nano-vllm-main，均已确认存在：

1. example.py → nanovllm/llm.py：外部如何发起生成。
2. nanovllm/engine/llm_engine.py：add_request、step、generate、结束判断。
3. nanovllm/engine/sequence.py、scheduler.py：本周只跟踪单请求状态，不实现新调度策略。
4. nanovllm/engine/model_runner.py：prepare_prefill、prepare_decode、run_model、run。
5. nanovllm/models/qwen3.py：embedding、decoder layers、Attention/MLP、logits。
6. nanovllm/layers/attention.py、rotary_embedding.py、sampler.py：缓存位置、RoPE、采样。
7. nanovllm/engine/block_manager.py：先知道何时分配和归还逻辑块，分页算法留到第五周。

跟踪日志字段至少包括：请求标识、阶段、输入位置数、输入 token ID、position、采样出的 token ID、当前输出数、停止原因。只截取少量步骤，日志运行与性能测量分开。

## 8. 运行边界与常见陷阱

- 先检查 GPU 当前负载，不能沿用第二周“GPU 1 空闲”的结论；不终止其他人的任务。
- 首次 batch=1、短提示词、max_tokens=16 或 32、tensor_parallel_size=1。nano-vLLM 首轮采用 enforce_eager=True 便于跟踪。
- 当前分支默认 gpu_memory_utilization=0.9，会主动分配较多 KV 空间。启动前明确配置适合本次小实验的预算、max_model_len 与批处理上限，并核查能分配到至少所需的块；不直接运行示例的默认大预算。
- 模型加载、编译、预热与实际生成分开记录。generate 整段耗时不是 TTFT；没有逐 token 时间戳时不报告 TPOT。
- BF16 权重文件大小、模型参数显存、有效 KV 字节、预分配 KV 和 nvidia-smi 进程占用不是同一个指标。
- 不直接覆盖两份源码中的用户改动。学习脚本与报告落在 /home/ubuntu/infer-engine/week3，引用外部源码时记录路径和版本；必要插桩先确认干净状态并保留改动边界。
- 模型能说出一句话只是链路跑通，不代表数值对拍、性能或请求生命周期验证全部完成。

## 9. 本周验收

- [ ] 记录实际加载的模型、tokenizer、nano-vLLM 源码路径和依赖版本。
- [ ] 能解释模板、token IDs、embedding、decoder、logits、选 token 和解码文本的关系。
- [ ] 跑通至少三条短提示词，保留输入/输出 IDs、参数和停止原因。
- [ ] 解释第一个输出 token 来自 Prefill，以及它何时写入 KV。
- [ ] 在真实参考模型内完成全重算与缓存增量逐步 logits 对拍。
- [ ] 跟踪 nano-vLLM 中一次请求从创建到结束，指出 KV 逻辑资源回收位置。
- [ ] 正确计算本地 28 层 GQA 模型的缓存字节，区分有效/预分配/进程显存。
- [ ] 汇总限制和未完成项，不将原生功能写成自己的新增优化。

达到这些条件后，Week 4 进入多个长短请求的调度与生命周期观察。若只剩跨框架高精度 logits 对拍未完成，可记录为进阶项；但不能跳过同模型缓存正确性和单请求链路理解。
