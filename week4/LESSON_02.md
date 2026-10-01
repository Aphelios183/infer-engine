# Week4 · Lesson2：调度、分页与Qwen3.5混合状态准入

前置：[Lesson1概念验收通过](LESSON_01_ASSESSMENT.md)。本课不启动GPU，先读源码和手推；实作按[适配主线](QWEN35_ADAPTATION.md)逐步接入。

## 1. 本课要解决什么问题

请求已经进入引擎，为什么不能立即全部执行？因为这一轮有计算预算，缓存有容量，还必须保证每个请求读写自己的状态。

Scheduler决定“谁执行、执行多少”；BlockManager管理“普通KV存在哪里、是否有可用块”；Runner把选择结果变成实际输入和地址。Qwen3.5还要回答“这个请求的conv/recurrent状态在哪里”。

阅读根目录：`/home/ubuntu/t1/nano-vllm-main/nano-vllm-main/nanovllm`。按以下顺序，先不钻CUDA kernel：

1. `engine/scheduler.py`：schedule → preempt → postprocess。
2. `engine/block_manager.py`：can_allocate → allocate → can_append → may_append → deallocate。
3. `engine/sequence.py`：num_tokens、num_cached_tokens、num_scheduled_tokens、num_blocks和block_table。

## 2. 三种数量不要混用

- num_tokens：请求持有的token总数，包含刚选出但还没forward的token。
- num_cached_tokens：此实现记账的已缓存前缀长度；前缀命中也可能直接增加它。
- num_scheduled_tokens：本轮计划处理的输入量，不是新生成token数。

schedule给计划；forward实际执行；postprocess更新cached并把scheduled归零。规划好了不等于执行成功。完整Prefill可能输入很多token但通常只选出一个下一token；中间分块不能作为完整prompt的输出交付。

## 3. 精读当前Prefill分支

源码关键条件：

```python
remaining = self.max_num_batched_tokens - num_batched_tokens
if remaining < num_tokens and scheduled_seqs:
    break  # only allow chunked prefill for the first seq
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

这里局部变量num_tokens是该请求尚需处理的量，不一定等于seq.num_tokens。

本轮尚未选择请求时，可以给第一条请求一个不完整分块；一旦已经选过请求，后续请求若放不下就停止选择，不再给它切小块。完整prompt被安排后，请求在forward之前就从waiting移到running，因此running不能直接翻译成“此刻一定Decode”。

只要选中了Prefill任务，函数就返回`(scheduled_seqs, True)`，本轮不进入下方Decode分支。这是当前读本的策略，不代表所有推理引擎如此。

## 4. 两个请求的预算实验

假设A的prompt为6个token、B为3个token，waiting=[A,B]，running=[]；每轮预算4，max_num_seqs=2，无前缀命中，块足够，二者不提前停止。

以下是按源码手推的预期，不是已运行的实验报告：

| 轮次 | 本轮输入量 | 完成postprocess后的结果 |
| --- | --- | --- |
| 1 | A处理4 | A.cached=4，A仍在waiting；不追加生成token；B未执行 |
| 2 | A处理剩余2 | 剩余预算2放不下B的3，且已选A，所以不选B；A.cached=6，A.num_tokens=7，A在running |
| 3 | B处理3 | Prefill优先返回，A本轮未Decode；B.cached=3，B.num_tokens=4 |
| 4 | A、B各Decode输入1 | 若资源足够且无新waiting，两者各推进一个输入并各选择下一token |

注意postprocess的这段代码：

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

部分Prefill时runner可能已经计算了采样结果，但这里忽略它，不追加到请求输出。不能据此说现有runner完全没做采样。

**自己解释：** 第二轮为什么宁可剩2个预算，也不让B先处理2个？这不是数学上不能，而是哪一条策略限制？

## 5. token预算不是显存预算

当前allocate遍历请求所需的块，可能为整个prompt预留普通KV块，不只为本轮实际处理量分配。计算4个token不意味着仅分配4个token的容量。

can_allocate返回前缀命中块数，失败返回-1；它不是普通布尔值。allocate才改变块引用和块表。deallocate减少引用，仅在引用归零时把块变为空闲，并清空当前请求block_table与cached计数；token_ids仍保留。

toy block_size=4、block_table=[5,2]、逻辑位置5：逻辑块1，偏移1，物理块2，slot=9。真实Config要求block_size为256倍数，toy数值仅用于独立推演。

Decode输入的是刚选出的最后一个token，位置为len(seq)-1。所以当len(seq)%block_size==1时，这个输入位于新逻辑块开头；may_append在forward前分配块。例：len=9、block_size=4，输入位置8，需要第三块。

## 6. 抢占不是“删除一条输出”

当前Decode分支资源不足时，可能抢占running尾部请求；preempt将请求放回waiting，释放它的块并把cached归零，但保留token序列以便重算。

这对Qwen3.5意味着：恢复时不能一边从完整token前缀重算，一边继续使用旧conv/recurrent状态，否则历史会被重复计入。首版先显式拒绝未验证的抢占恢复；真正支持时，必须规定状态清理/重建与KV释放的一致边界。

## 7. 对照你上传的正式vLLM

根目录：`/home/ubuntu/vllm-main/vllm-main`，不是外层目录；该源码版本未确认。

读 `vllm/v1/core/sched/scheduler.py::schedule`：开头解释用num_computed_tokens追上num_tokens_with_spec；随后构造每请求num_scheduled_tokens和共享预算，不返回nano式整个batch单一is_prefill。

“待计算量=目标长度-已计算长度”是入门理解。实际公式还有output placeholders，随后受token/input预算、最大长度及混合状态对齐等限制。不要把简化式当完整实现，也不要把调度时推进的记账当成GPU已经完成。

然后对照 `vllm/v1/engine/core.py::step` 与Scheduler.update_from_output：非阻塞提交后可准备grammar约束，等待执行结果再结算。CPU/GPU是否真的重叠要用测量证明，grammar不是训练正则化。

本课只比较两个问题：如何表达每请求工作量？资源不够时在哪里拒绝/等待/抢占？不要求现在读完生产调度器。

## 8. 本课如何直接服务Qwen3.5适配

普通Attention请求的准入检查不能直接覆盖混合模型。以后执行一个新请求，至少同时满足：本轮计算预算、Full Attention KV资源、linear状态slot。

示例：KV还有100块，但linear状态slot已满。此时不能只按can_allocate成功就运行，也不能称GPU显存全满；可能只是我们的状态池容量不足。

拟定契约（尚未实现）：

1. 先检查不支持的功能和永久无效配置，不改变任何资源。
2. 检查并预留KV资源和linear slot；任一失败，回滚本次已经拿到的资源。
3. 两类资源均成功才提交请求映射与调度计数。
4. 首版不完整前缀复用、分块Prefill、抢占恢复、Graph均要有实际guard，不靠注释“假装关闭”。

这里不是要求立刻写完整状态池。先完成CPU准入契约及失败用例；Lesson4实现状态池，Lesson5接通真实生命周期。纯CPU模拟通过不等于模型适配通过。

## 9. 本课验收与下一步

在 `week4/notes/scheduler_trace.md` 记录手推；实现阶段建立独立CPU测试。验收要求：

- 能解释第4节每轮队列、scheduled、cached和新输出的变化。
- 能解释连续批处理、分页、前缀复用的不同职责。
- 能解释一次块边界分配、共享块引用归零及资源耗尽。
- 给出Qwen3.5准入失败时应保持不变的字段和资源。
- 对照正式V1指出一个真实机制差异，不只说“更复杂”。

第一道题：沿用第4节的A=6、B=3，但将预算改成5。第一轮给谁处理几个token？第二轮A还需几个，B能否在同轮完成Prefill？请同时写出第二轮postprocess后A、B的num_tokens与num_cached_tokens。
