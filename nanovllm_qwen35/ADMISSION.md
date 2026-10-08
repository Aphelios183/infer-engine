# 下一部分：KV与linear状态联合准入

实现：`qwen35_adapter/admission.py::AdmissionController`；测试：`tests/test_admission.py`。后续已接入独立CPU [HybridScheduler](SCHEDULER.md)，新增reserve_decode用于块边界扩容及异常回滚；try_admit只负责初次准入。

## 已实现

连接副本中的真实Sequence、BlockManager与CPU StateManager，不再用模拟KV分配器。管理器必须由协调器串行、独占调用，不支持外部同时修改或线程并发。

```python
controller = AdmissionController(block_manager, state_manager, max_prefill_tokens=budget)
slot = controller.try_admit(seq, token_budget=remaining)
if slot is None:
    # 暂时没有足够资源或本轮剩余预算，让请求等待。
    pass
else:
    # 资源预留成功；slot=0也成功。
    # 下一部分由调度器负责入批、设置scheduled计数与队列迁移。
    pass
```

当前只接受未执行、无块表、无缓存计数的新请求。prompt超过单轮最大预算或总KV容量会报错，而不是永远等待；本轮剩余预算不足则暂缓。未支持分块、恢复和前缀共享，使用`BlockManager.allocate(seq, num_cached_blocks=0)`，不采用普通Full KV的命中长度。

## 事务顺序

1. 校验请求状态、block_size、容量和预算，永久无效请求直接拒绝。
2. 记录BlockManager元数据与请求块表/计数快照。
3. 分配普通KV块，不推进num_cached_tokens和num_scheduled_tokens。
4. 调用StateManager.allocate，初始化属于请求的两类状态。
5. 两者成功才登记协调器的请求所有权并返回slot。
6. 若状态池满或任一步抛异常，恢复KV元数据与请求计数；StateManager保证初始化失败不遗留所有权。

`release(seq)`校验请求对象身份后归还两类资源，拒绝伪造同ID对象或重复释放。它不设置FINISHED、不取消在途GPU任务；调用者必须确认执行已经停止。目前已有CPU调度器接线，仍没有真实模型生成。

## 为什么不只调用deallocate就说完全回滚

普通分配可能淘汰空闲块保存的prefix hash；普通deallocate又会改变空闲队列顺序。要满足本课“失败后元数据快照相同”，本版保存并恢复空闲顺序、used集合、hash索引、块ref_count/hash/token列表以及请求块表/计数。没有复制实际KV张量。

这是O(总块数)的CPU元数据快照，优先验证正确性，不应当作最终性能设计。后续可换成预留日志/局部undo记录，再测开销。本版不保护并发读写，也不回滚执行中的模型状态；未归属state slot可能已部分清零，下次分配会重新初始化。

## 测试与源码改动

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

10个联合准入测试加此前24个测试，共34项通过。包括：slot0成功、状态池满后撤销KV、KV不足、KV分配中途异常、state初始化异常、禁止不完整prefix复用、恢复已淘汰的空闲prefix元数据、预算/永久超容量、非新请求拒绝、释放与邻居隔离。

测试用block_size=4只作CPU机制验证，不经过生产Config的256倍数限制。为了直接导入真实Sequence/BlockManager，副本`nanovllm/__init__.py`改为按需加载LLM：导入资源模块不再连带导入GPU Runner；`from nanovllm import LLM`才加载引擎。原来源目录不改；Qwen3.5原生执行guard仍保留。

## 距离真实适配完成还缺什么

| 部分 | 当前状态 | 下一交付 |
| --- | --- | --- |
| 配置/层映射 | CPU已验证 | 后续核对实际权重与后端dtype |
| StateManager | CPU同步已验证 | GPU池、句柄/在途执行安全、真实布局 |
| 联合准入与Scheduler | CPU队列、预算、结束/失败回收已测试 | 接Runner输入与执行边界 |
| Runner元数据 | CPU输入准备已消费调度批次 | 接执行后端、分层模型状态和GPU安全生命周期 |
| Qwen3.5模型与权重 | 未接通 | 模型选择、文本权重/packed投影、RoPE/gate、Full和GDN执行 |
| 完整生成 | 未接通 | EOS/采样、停止/失败回收、真实单请求生成 |
| 正确性 | Week3仍diagnostic | 相同输入logits对照、dtype与容差批准、隔离回归 |
| 优化功能 | 未开放 | 正确性后再逐项做分块、prefix、Graph与性能分析 |

首个“适配完成”的目标限定单卡eager纯文本正确推理，不要求先做多模态、量化或PD分离。M1参考后端桥接和M2原生执行必须分别标注。Scheduler与CPU Runner输入准备已接通，下一part是执行接口与真实模型/权重接入，不直接跳到性能测试。
