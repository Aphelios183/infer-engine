# Lesson2 CPU契约测试

## 目的与边界

把课堂口头规则变为可重复失败/通过的断言。`hybrid_contract.py`是独立教学模型，`test_hybrid_contract.py`测试它；不导入或修改nano-vLLM，不加载Qwen3.5、不用GPU，也不证明真实模型适配成功。

真实模型有逐层conv/recurrent张量，本实验用两个短列表代表状态，使用 `new = 0.5 * old + token` 作为残留检测器。这个公式不是Gated DeltaNet。这里的processed_tokens只表示模拟forward处理量，不是生产系统包含prefix命中/调度预推进的完整计数。

## 测试设计：先布置现场，再触发事件，最后检查不变量

| 现场与事件 | 必须检查 | 捕捉的错误 |
| --- | --- | --- |
| 两类资源足够，接纳A | 同时取得KV和slot，processed仍为0 | 分配被误认为执行 |
| slot满但KV有余量，C到来 | 返回等待，整个状态不变 | 覆盖其他请求或泄漏KV |
| slot有余量但KV不足 | 预留slot退回，C无所有权 | 半完成分配 |
| KV分配或初始化抛异常 | 异常传播，已取得资源全部退回 | 只处理普通失败、遗漏异常 |
| 无空闲slot，已有A继续 | slot和底层列表地址不变，值更新 | 每token重复分配状态 |
| 只有Full KV的前缀命中 | 在修改状态前拒绝 | 错误跳过linear历史 |
| A结束，C复用其slot | 两类状态清零；C首步等于从零执行 | 只清计数或只清一种状态 |
| batch从[A,B]改为[B,A] | 按请求读取和更新，各自值正确 | 把batch下标当所有权 |
| 释放A，C重试 | B的状态不变，C可以接纳 | 误释放邻居、等待不能恢复 |
| 重复接纳/释放、请求超过总容量 | 明确拒绝且无副作用 | 重复归还、永久无法调度却死等 |

使用不同哨兵值（A=10、B=20）让串状态容易被观察；失败路径比较完整资源快照，不仅检查返回False。复用测试故意保留释放后的脏数据，确保清零确实发生在新请求使用之前。

## 运行

在服务器执行，不需要空闲GPU：

```bash
cd /home/ubuntu/infer-engine
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s week4/test -p 'test_*.py' -v
```

共12个测试。实际运行结果见同目录 `RESULTS.md`。

## 如何证明测试能抓到错误

在内存中临时把 `_reset_slot` 替换为空操作，再运行复用测试，应失败；恢复后应通过。该故障注入不修改源码，验证测试不是只跑正常流程。其他学习练习：删掉slot回滚、按slot编号而非request顺序返回batch，各预测哪个测试先失败。

## 接入nano-vLLM时仍需验证什么

本实验假定单线程串行准入；不模拟分配函数内部“已修改但未返回”的失败，不证明并发原子性。fake_forward不会扩展KV、不执行真实注意力，也没有GPU异步生命周期。

后续把这些断言迁移到真实scheduler/state manager接口，再补上：每层实际张量与dtype、真实KV块引用、块边界增长、采样计数、forward失败清理、取消及在途执行、功能guard、实际模型数值对拍。前缀测试只验证首版拒绝规则，不证明完整混合前缀复用已实现。
