# Runner输入准备：消费调度批次

实现：`qwen35_adapter/runner_inputs.py`。测试：`tests/test_runner_inputs.py`。

## 当前接线

真实Sequence → HybridScheduler.schedule → ScheduledBatch → RunnerInputBuilder.prepare → RunnerInputs。

这一步接通CPU输入准备，不调用模型、不传CUDA、不设置上游全局context、不采样、不自动结算，也不新增KV块或状态slot。未来执行器要消费这份对象并适配算子接口，不能把它直接当作原ModelRunner或FlashAttention的参数包。Qwen3.5执行guard仍保留。

```python
builder = RunnerInputBuilder(scheduler)
batch = scheduler.schedule()
if batch is not None:
    inputs = builder.prepare(batch)
    # 下一阶段在这里接模型执行和采样。
    # 同步执行完成后，scheduler.postprocess(batch, token_ids)
```

只能准备当前尚未结算的batch；过期、伪造、属于别的调度器的batch拒绝。输出是新分配的CPU张量，不与请求列表或状态池共享数据。dataclass的冻结不等于张量只读，调用方必须将其作为只读输入；不能在请求释放后继续使用旧输入执行。

## 每个字段负责什么

| 字段 | 布局/类型 | 用途 |
| --- | --- | --- |
| request_ids | tuple，按batch顺序 | 后续输出归属 |
| input_ids | [本轮token总数]，int64 | 仅本轮需要执行的token |
| positions | 同上，int64 | 每个请求内部的逻辑位置，不是拼接下标 |
| cu_seqlens_q | [请求数+1]，int32 | 本轮Q累计边界 |
| cu_seqlens_k | [请求数+1]，int32 | 各请求有效KV长度的累计边界，不是物理地址 |
| context_lens | [请求数]，int32 | 包含本轮输入后的有效KV长度 |
| slot_mapping | [本轮token总数]，int32 | 新K/V的物理写入位置 |
| block_tables | [请求数, 本批最大块数]，int32 | 历史KV的物理块表，短行补-1 |
| state_slots | [请求数]，int32 | 每个请求的linear状态位置 |
| last_indices | [请求数]，int64 | 本轮packed输出中每请求的最后位置 |
| max_seqlen_q/k | Python int | 本轮最大Q/有效KV长度 |

所有张量显式device=cpu。-1是块表补齐标记，绝不能作为物理块访问；读取历史必须服从context_lens。当前通用元数据在Prefill也保留块表，Decode也保留累计边界，后续后端适配时只传需要的参数。

## 已测试的课堂例子

完整Prefill顺序[B,A]，B=[51,52]使用物理块2，A=[11,22,33]使用物理块5，状态slot分别0和1：

```python
input_ids    = [51, 52, 11, 22, 33]
positions    = [0, 1, 0, 1, 2]
cu_seqlens_q = [0, 2, 5]
cu_seqlens_k = [0, 2, 5]
slot_mapping = [8, 9, 20, 21, 22]
state_slots  = [0, 1]
last_indices = [1, 4]
```

选出B的53和A的44后，Decode顺序换成[A,B]：

```python
input_ids    = [44, 53]
positions    = [3, 2]
context_lens = [4, 3]
slot_mapping = [23, 10]
state_slots  = [1, 0]
cu_seqlens_q = [0, 1, 2]
cu_seqlens_k = [0, 4, 7]
last_indices = [0, 1]
```

继续一轮后A输入45，位置4，新增物理块7，写slot28；A只读[20,21,22,23,28]这5个历史位置，不读预留块的剩余槽。

## 检查与失败语义

准备阶段核对：请求仍归准入器拥有且处于running；state slot与请求映射一致；本轮数量与scheduled相同；输入区间恰好覆盖待处理token；块表长度、物理块号、used集合和引用计数有效；禁止共享/重复块；新token非负整数。

仅支持全新完整Prefill和每请求1个token的Decode，不偷偷启用prefix/chunk。每个请求block_size必须与池一致，last_token必须与token_ids末尾一致。

prepare失败不改计数、不回收请求、不清状态，错误交给调用方。若仅准备阶段失败且没有任何forward开始，可明确确认空闲后fail_batch；不能把这个CPU确认方式直接用于GPU异步执行。测试中的元数据损坏用于注入错误，不是允许业务代码随意修改资源表；损坏资源表要先修复/隔离后再回收。

## 验证

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_runner_inputs.py' -v
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_*.py'
```

2026-10-08：新增13项、全套65项测试通过。覆盖Prefill边界、Decode重排、跨块/补齐/历史范围、CPU dtype、无状态副作用、输出不别名、过期/伪造batch、slot错配、非法/重复/缺失块、错误scheduled、拒绝隐式分块，以及调度→准备→合成EOS→回收的控制链。合成EOS不是模型生成。

下一阶段是执行接口与Qwen3.5模型/权重接入：明确后端消费方式、真实状态dtype/布局、GPU安全生命周期，再做真实单请求对拍。当前不声称GPU Runner或模型已经适配完成。
