# Week 4 Lesson 1：一个请求如何走到第一次 forward

本课先读真实代码，不运行构造LLM的命令：当前构造函数会初始化NCCL、加载模型、预热、分配KV，默认还捕获CUDA Graph，且模型类仍为Qwen3。

## 1. 先纠正 Week 3 收尾答案

1. prompt18、生成4个且因长度停止：1次Prefill、3次Decode，已处理21个token，正确（假设无分块Prefill）。
2. input_ids表示token身份；positions表示它处于序列中的位置；cache保存此前计算状态。单个长度不能替代内容和每请求映射。
3. Full Attention保留历史K/V；Linear Attention保留conv/recurrent state。短期/长期仅为比喻。
4. 共享模型权重不等于共享请求可变状态。严格管理的不可变前缀块可复用，但需引用计数与写入隔离。
5. batch重排要保持token、位置、KV块表、conv/recurrent slot、长度、采样配置、输出request_id对应。可改变映射索引，不必物理搬移所有缓存。

## 2. 真实阅读顺序

源码根目录：/home/ubuntu/t1/nano-vllm-main/nano-vllm-main。

| 顺序 | 文件与入口 | 本次只回答 |
| --- | --- | --- |
| 1 | nanovllm/llm.py: LLM | 谁继承谁？ |
| 2 | engine/llm_engine.py: generate/add_request | 字符串在哪里变成IDs？是否自动套chat template？ |
| 3 | engine/sequence.py: Sequence.__init__ | 请求ID、状态、token列表、已缓存长度存在哪里？ |
| 4 | engine/scheduler.py: add/schedule | 谁决定这一轮运行哪些请求？ |
| 5 | engine/llm_engine.py: step | 调度和执行如何接起来？ |
| 6 | engine/model_runner.py: run | 谁准备输入、调用模型、采样？ |
| 7 | engine/scheduler.py: postprocess | 谁append token、检查停止、回收块？ |

## 3. 核心控制链（不是生产vLLM进程图）

```text
LLMEngine.generate
  → add_request → Sequence → scheduler.add → waiting
  → 循环 step
      → scheduler.schedule → seqs, is_prefill
      → model_runner.call("run", seqs, is_prefill)
          → prepare_prefill / prepare_decode
          → run_model → 模型forward与compute_logits
          → sampler → token_ids
      → scheduler.postprocess
          → 更新已处理计数 → append_token → 检查停止 → 释放块
  → 按seq_id收集输出 → tokenizer.decode
```

实际源码片段（llm_engine.py 的 step）：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

调度器不做模型矩阵计算，ModelRunner不直接决定waiting队列先挑谁，Sequence也不是整个模型的GPU缓存。

## 4. 本地实现中容易误解的地方

- add_request对字符串调用tokenizer.encode，没有在这里自动应用chat template；适配时需要明确输入是否已模板化。
- LLMEngine将tokenizer.eos_token_id赋给config.eos。Week3记录过模型generation EOS与tokenizer EOS不同，不能直接沿用而不验证。
- Sequence.num_tokens包含已追加但可能尚未forward的最后一个token；num_cached_tokens表示该实现已计算的数量。
- postprocess会先增加num_cached_tokens，完成Prefill后再append采样token；分块Prefill未完成时不append。
- 停止后deallocate会把num_cached_tokens清零并清空block_table，所以应注明trace取在释放之前还是之后。
- 源码中已有中文注释；以实际语句判断，不把注释当运行证据。

## 5. 正式 vLLM 对照

阅读[对照卡第1节](FRAMEWORK_COMPARISON.md)。正式V1的职责跨输入侧、Engine Core和GPU Worker分布；本地的三行step是学习抽象，不是正式V1所有进程的实际调用栈。
写出“调度、执行、回收”三个职责在两边各由谁承担，暂不追所有RPC细节。

## 6. 第一题与本课验收

看上面三行step，不看答案，回答：

1. 哪一行决定本轮运行哪些请求？
2. 哪个阶段产出新的token ID？
3. 哪个阶段把token追加到Sequence并决定是否释放缓存？

然后自己在 week4/notes/request_lifecycle.md 画prompt=[11,22,33]、生成2个token的两轮状态表：记录phase、num_tokens、num_cached_tokens、num_scheduled_tokens、status、block_table是否为空。notes文件由你后续创建，本轮不伪造已完成作业。
本课通过标准：能定位上述入口，并说明采样与写入缓存不是同一件事。
