# StateManager：CPU同步接口与验证

实现：`qwen35_adapter/state_manager.py`。测试：`tests/test_state_manager.py`。

这是适配代码中的真实张量状态池管理原型，不再是week4/test中的列表模拟；但仍未接到引擎，没有Qwen3.5计算，也不支持GPU或并发执行。非CPU设备会在构造时拒绝。

## 已确认的接口

```python
allocate(request_id)  # 成功返回int（0合法）；池满None；重复请求ValueError
lookup(request_id)    # 返回已有slot；未知请求KeyError；不清零
release(request_id)   # 撤销所有权并归还slot；未知/重复释放KeyError
```

request_id接受整数（对应nano Sequence ID）或非空字符串，拒绝bool与其他类型。调用方应使用生命周期唯一ID；本版没有generation句柄，不能让异步旧回调以同一ID操作新请求，也不能长期保存裸slot当作所有权凭证。

构造函数接收预先创建的conv_pool与recurrent_pool，两者前两维为 `[linear层数, 请求容量]`；分别要求rank4与rank5、正维度、连续浮点张量、无需梯度、独立存储。dtype由调用方显式指定，本类不从模型默认dtype猜测，也不改变dtype。真实模型形状由配置契约和未来构建器负责对接，本类只验证通用池结构。

## allocate的顺序

1. 校验ID并拒绝重复请求。
2. 空闲集合为空则返回None，任何内容都不改变。
3. 暂取一个slot。
4. 仅清零 `conv_pool[:, slot, ...]` 和 `recurrent_pool[:, slot, ...]`。
5. 初始化成功后才登记request_id到slot映射。
6. 初始化失败时把slot退回原位置并传播异常，不提交所有权。

异常回滚保证资源记账不变；未归属slot的字节可以已被部分清零，下次分配必须重新初始化。不会假装恢复模型历史。已有请求的切片不能被碰触。

release不重新分配张量，也不清整个池；保留的脏字节在下次allocate时清除。CPU同步调用方必须确保该请求不再使用状态，再释放。将来GPU版必须额外处理在途执行和安全复用，不能删掉设备guard就宣布支持。

## 阅读与运行

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_state_manager.py' -v
```

先读 `test_zero_is_success_and_both_slices_are_initialized`，再读 `test_reuse_clears_all_layers_without_touching_neighbor`，最后读部分初始化失败回滚测试。历史CPU测试汇总已按要求归档；当前GPU诊断见BACKEND_AUDIT.md。

代码和新增测试由助手实现，不自动代表学生独立实作通过。下一步由学生解释“为什么初始化后才提交owners”；随后设计StateManager与BlockManager的联合准入，而不是直接加载GPU模型。
