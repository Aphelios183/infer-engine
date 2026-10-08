# HybridScheduler：队列、预算与结束/失败回收

实现：`qwen35_adapter/scheduler.py`。测试：`tests/test_scheduler.py`。

## 已接通的边界

HybridScheduler → AdmissionController → 真实nano BlockManager + CPU StateManager。
复用真实Sequence和SequenceStatus，不替换原scheduler.py，尚未接到LLMEngine/ModelRunner。Qwen3.5执行guard保持不变。

本版为单线程、同步、单在途batch的CPU控制链。测试直接提供token结果或注入失败，未执行模型、CUDA或真实采样。

## 调用顺序

```python
scheduler.add(seq)
batch = scheduler.schedule()
# batch.sequences / is_prefill / state_slots / input_counts
# 调用方在这里执行同步模型与采样；本阶段尚无真实Runner接线。
scheduler.postprocess(batch, token_ids)
```

每个batch只能结算一次。未结算时再次schedule、重复结算、用伪造batch结算均拒绝。队列、资源和sequence由调度器独占；调用者不得在执行期间自行修改或释放请求。

- add校验新请求和永久容量限制；不可能完整Prefill的prompt直接拒绝，不死等。max_tokens=0直接结束，不运行Prefill。
- schedule先选可执行的running Decode请求，每请求一个输入，共享token预算，选中者轮转至队尾。没有可执行Decode时才选完整Prefill；扫描waiting允许跳过本轮放不下的请求，不分块、不混合阶段。
- max_num_seqs限制活跃请求数及单批请求数。准入成功才waiting→running、设置scheduled；此时尚未forward。
- postprocess先验证整批输出数量和类型，再推进cached、归零scheduled、追加token。EOS优先于长度停止，尊重ignore_eos。
- 结束时归还两类资源、移出队列；Sequence保留输出。completed保存reason、token快照及释放前已结算的processed_tokens。失败forward可能部分写入，但不算作成功处理历史。

## 策略与原版差别

本版选择Decode优先的同阶段batch，不是上游nano的Prefill优先，也不是正式V1的统一token工作量/混合阶段调度。该限制便于验证首版状态边界，不是已证明最优的策略。

Decode优先可能增加waiting的TTFT；扫描跳过大请求可能影响公平性。有限max_tokens不是生产SLO保证，后续需测量再改策略。

## 缺块、无进展与失败

Decode沿用原state_slot，必要时reserve_decode追加KV块；追加异常会先回滚本次块操作。缺块时不选该请求，不推进计数，不释放已有历史，允许其他请求执行并释放资源。

若还有请求但整轮无可执行任务，抛NoProgressError，保留资源和队列。调用方必须明确取消/失败某个请求或调整容量策略，不能捕获后无限重试。本版不自动抢占或偷偷清状态重算。

```python
# 同步执行失败后，且已经确认执行停止：
scheduler.fail_batch(batch, error, execution_complete=True)
```

fail_batch终止整批可能被部分修改的请求，不从脏状态重试。execution_complete是调用者的承诺，不是GPU事件检查；默认未确认时拒绝回收。本版不适用于GPU异步执行。

空闲waiting/running请求可通过fail(request_id,error)或cancel(request_id)终止。pending batch成员不能单独释放，需先结算或安全终止整批。沿用SequenceStatus.FINISHED表示终态，completed.reason区分eos/length/failed/cancelled，不能把所有FINISHED当作生成成功。

输出数量错误或非法token可能发生在模型已经写状态之后，因此整批失败回收，不静默zip截断、不留下脏状态重试。

## 验证与阅读顺序

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_scheduler.py' -v
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

新增18项Scheduler测试通过，全套52项通过。先读两轮生命周期测试，再读缺块等待邻居结束、无进展检测、部分forward失败及资源复用测试。源码和测试由助手实现，不自动代表学生独立实作通过。

限制：CPU元数据快照O(块数)，无线程安全/句柄generation，completed和已用ID集合暂无生产级回收，无真实GPU事件同步，无prefix/chunk/preemption。

下一part：Runner消费ScheduledBatch，构造input_ids、positions、slot_mapping、block_tables、长度、state_slots及层映射，验证顺序与异常路径。然后接GPU状态池和真实Qwen3.5模型/权重。
