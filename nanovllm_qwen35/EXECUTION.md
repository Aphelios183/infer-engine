# 执行接口：从输入准备到结算

实现：`qwen35_adapter/execution.py`；测试：`tests/test_execution.py`。
2026-10-08：11项新增测试，全套76项通过（0.203s）。这是CPU合成后端控制链，不是Qwen3.5数值验证。

## 阅读顺序与职责

1. `CPUBackend`：后端契约。构造时绑定存储；forward接收RunnerInputs，返回`[本轮token数, hidden_size]`。后端负责按KV地址、state_slots访问存储，不能修改调度计数或归还资源。
2. `CPUExecutionRunner.run`：prepare后调用forward，按last_indices取每个请求最后一行隐藏状态，再调用compute_logits，得到`[请求数, vocab_size]`的原始分数，不是概率。
3. `CPUSampler.sample`：接收logits与按请求排序的temperature，返回`[请求数]`的int64 token。测试ToySampler故意使用argmax，只验证接口；不是正式nano采样器，也没有实现temperature采样。
4. Runner校验结果后调用Scheduler.postprocess。只有这里推进计数、追加token、判断结束和释放。run已完成结算，调用者不能再postprocess一次。

```python
runner = CPUExecutionRunner(scheduler, backend, sampler)
batch = scheduler.schedule()
if batch is not None:
    result = runner.run(batch)
    # result按request_ids返回token_ids；不再次结算！
```

Prefill输入[B:51,52; A:11,22,33]，forward输出5行；last_indices=[1,4]，只将对应52与33的隐藏状态送入logits头。Decode每请求只有一行。最后选出的EOS/长度终止token不自动再forward，也就不写入缓存。

## 执行与资源边界

- prepare失败：没有开始执行，保留pending batch与资源；先检查/修复元数据，不能盲目释放损坏资源表。
- forward、logits或sampler失败：历史可能部分改变。当前严格同步CPU后端退出即无在途工作，整批标记失败并归还资源，不回滚历史、不在脏状态重试。
- postprocess自身失败：不执行第二次自动回收，避免部分结算后的重复释放；应停止并检查管理器一致性。
- 后端和采样器必须声明`execution_mode='cpu_sync'`，且异常退出也不能留下后台工作。声明是可信实现契约，不能检测后端偷偷启动CUDA/线程。严禁把异步GPU后端仅改个标签接进来。
- 所有结果必须为CPU张量；检查形状、浮点结果有限性、采样dtype和词表范围。这里要求未加采样mask的原始logits有限；未来mask/约束解码需要独立契约。

没有创建生产KV池、传CUDA或更新Qwen3.5执行guard。StateManager仍是CPU原型。未接入原LLMEngine；应用必须串行调用，不能绕过runner并发操作调度器。

## 测试为什么这样设计

ToyBackend使用单标量/token的假KV数组，按slot_mapping写入；按state_slots原地累加小型conv/recurrent池。它不计算Attention或GDN，只让错误地址、重复清零和请求串位可以被观察。生产层索引映射与真实张量布局将在模型后端中消费，本次没有伪装成已经实现。

覆盖完整Prefill→Decode→长度结束、last_indices、请求重排、续算状态保留、EOS不送入forward、部分状态更新后失败、NaN/错误logits形状、采样异常/非法shape/越界token、过期batch、准备失败保留资源、拒绝非同步CPU后端。

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_execution.py' -v
```

## 下一部分

核查本机Qwen3.5参考模型的权重命名、Full/GDN层调用与状态dtype，建立真实模型后端。参考Transformers桥接和原生nano实现分开验收，不能用ToyBackend测试通过替代真实模型正确性。GPU接入前必须另建同步/事件与失败隔离机制。

课堂问题：forward已修改A、B状态，但sampler抛异常，为什么不能只重新执行sampler之前的整轮forward？请结合“计数尚未提交”和“张量已改变”解释。
