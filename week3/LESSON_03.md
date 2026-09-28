# Lesson 03：混合缓存的状态、存储与生命周期

当前入口：先读第1–3节并运行CPU实验。工作目录为 /home/ubuntu/infer-engine。
本课沿用Lesson 2的单请求循环，不增加调度器、多卡或优化算子。

## 1. 从“会生成”走向“知道状态在哪里”

你已理解：新选出的token不一定已经经过模型；每轮forward才更新对应状态。
现在要分别观察两类层：

| 层类型 | 保存什么 | Decode时主要观察什么 |
| --- | --- | --- |
| 8层 Full Attention | keys、values | 历史长度增长，历史部分是否保留 |
| 24层 Linear Attention | conv_states、recurrent_states | 形状可能固定，但数值随着输入更新 |

先前两次forward的实测，不是本课新实验：
- Full层K/V：(1,4,17,256) → (1,4,18,256)。
- Linear层conv_states：(1,8192,4)。
- Linear层recurrent_states：(1,32,128,128)。
- 当时这些状态实测为BF16，而不是仅凭配置猜测的FP32。

不要从固定形状推导“没保存历史”，也不要从数值变化推导“计算一定正确”。
数值对拍留到Lesson 4。

## 2. 引用不是快照

下面是引用同一个张量：

```python
before = cache.layers[0].recurrent_states
```

如果后续原地更新该张量，before看到的内容也会变化。
如果实现改为替换张量，旧引用可能保留旧值；正因为更新方式可能不同，不能依赖这种偶然行为做实验。

保存独立数值快照：

```python
before = cache.layers[0].recurrent_states.detach().clone()
```

detach()去掉梯度跟踪关系，但单独使用仍共享存储；clone()才复制数据。
inference_mode中通常无需为梯度使用detach，这里保留它便于明确表达“诊断快照”。
不要用 .data 绕过张量管理。

## 3. 先用CPU验证引用与复制

从项目主目录执行；此实验不加载模型、不使用GPU：

```bash
CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B - <<'PY'
import torch

state = torch.tensor([1.0, 2.0, 3.0, 4.0])
alias = state
detached = state.detach()
snapshot = state.detach().clone()

state.add_(10)  # 下划线表示原地修改
print("state:", state.tolist())
print("alias:", alias.tolist())
print("detach only:", detached.tolist())
print("snapshot:", snapshot.tolist())
print("快照最大差异:", (state - snapshot).abs().max().item())

assert alias.data_ptr() == state.data_ptr()
assert detached.data_ptr() == state.data_ptr()
assert snapshot.data_ptr() != state.data_ptr()
assert torch.equal(snapshot, torch.tensor([1., 2., 3., 4.]))
assert torch.equal(alias, torch.tensor([11., 12., 13., 14.]))
assert (state - snapshot).abs().max().item() == 10.
print("引用/快照实验通过；不代表真实模型状态正确性通过")
PY
```

这证明的是PyTorch内存语义，不是Qwen3.5的缓存实现已经对拍通过。

第一个问题：为了观察而额外分配的snapshot，应该计入“模型自身缓存”吗？
答案：不计入模型缓存统计，但如果快照放在GPU上，确实会占用进程显存，必须单列诊断开销。
所以做性能/峰值显存测量时，不能混入大量clone后仍声称测到的是原始模型开销。

## 4. 三种内存数字不要混为一谈

1. 逻辑张量字节：tensor.numel() * tensor.element_size()。
2. 唯一底层存储字节：对共享的storage去重后求和。
3. CUDA进程/分配器占用：还包含权重、临时张量、诊断快照、保留块等。

例如：

```python
base = torch.zeros(8, dtype=torch.float32)
view = base.view(2, 4)
```

base和view各自逻辑大小都是32字节，加起来64字节；但它们共享同一块32字节存储。
统计时可用(device, storage.data_ptr(), storage.nbytes())识别同一个底层存储。
tensor.data_ptr()可能因视图偏移而不同，不能拿它直接当通用storage去重键。
不同storage的字节总和仍不等于memory_reserved()或nvidia-smi占用。

## 5. 准备真实状态实验（本课后续执行）

以下保留实验思路与片段；已完成的真实结果见 [本课报告](LESSON_03_RESULTS.md)。片段本身不是独立脚本。
必须先核查当时GPU负载，不沿用Lesson 2的GPU5空闲结论。

### 实验A：Full Attention增长与历史保留

1. 对一个短提示词Prefill，记录实际层索引、shape、dtype、有效长度。
2. 选择一个Full层，例如原始第3层，复制更新前K/V。
3. Decode一个非EOS token。
4. 读取更新后的层状态；长度应从P到P+1。
5. 对比after[..., :P, :]与before，而不是比较整个新旧张量的不同形状。
6. 如果不一致，保留差异并定位，不通过更改阈值隐藏问题。

### 实验B：Linear状态是否更新

```python
# 前提：cache来自成功的Prefill；在一次Decode前保存快照。
before = cache.layers[0].recurrent_states.detach().clone()

# 这里运行一次Decode，得到updated_cache。

after = updated_cache.layers[0].recurrent_states
same_shape = before.shape == after.shape
finite = bool(torch.isfinite(after).all().item())
max_abs_change = (after.float() - before.float()).abs().max().item()
```

记录same_shape、finite和max_abs_change，conv_states也按同样方式观察。
先检查shape一致再做减法。float()便于计算差异，但会产生额外诊断张量。
差异大于0说明这个样本的数值发生变化，不说明数学结果正确。
若差异为0，也不能立刻判定缓存坏了；检查输入、更新方式、精度和是否取错层，再增加受控样本。
不要要求每层、每个元素、每一步都必须变化。

观察张量时不要强行转换模型保存的状态dtype。状态的实际dtype、shape按读取结果记录。

## 6. 不同请求的状态必须独立

A请求结束，再开始B请求：

- B第一次forward必须cache=None、processed_tokens=0。
- 不把A的cache作为默认参数、全局变量或模型封装中的隐藏请求状态传给B。
- 比较“先A后B”与“B单独运行”，必须固定B的输入IDs和设置。
- 异常退出后，下一个请求也从新状态开始。
- 不仅检查长度，还要通过后续数值对拍确认没有污染。

本课先观察生命周期；正式数值验收需要Lesson 4固定前缀和预先声明的容差。
释放Python引用不等于nvidia-smi立即归零，CUDA分配器可能保留已申请空间。
本课不要求在每一步调用empty_cache()。

## 7. 记录方式与范围

只读观察器已实现为 [cache_inspection.py](cache_inspection.py)，测试为 [test_cache_inspection.py](test_cache_inspection.py)。它只记录元数据，不为你保留数值快照。
观察器的返回值只能是JSON可保存的数字/字符串/列表，不保存cache或GPU张量引用。
诊断快照短时持有，比较后释放；不要把所有层每一步的clone长期塞入trace。

报告至少记录：
原始层索引、层类型、状态名称、shape、dtype、逻辑字节、去重storage字节、已处理token数；
缺失/未初始化为null，不伪造“0字节已验证”。

## 8. 本课练习与验收

先完成前3节，再逐项推进：

- [ ] 解释直接赋值、detach()与clone()的区别。
- [x] 区分模型缓存和诊断快照的显存占用（已答对）。
- [ ] 手算8个FP32元素及其view的逻辑字节与唯一存储字节。
- [x] 实现只读状态观察器与共享存储单元测试（10项通过）。
- [x] 实测全部层元数据，并抽查第3层K/V历史保留与第0层Linear数值变化；不泛化为全部层数值正确。
- [ ] 检查请求切换与异常后的状态重置（待做）。
- [ ] 将结论写成“观察到了什么”，不提前声称全重算/缓存数值正确性通过。

观察器与本课真实观察已完成，完整结果见 [报告](LESSON_03_RESULTS.md)。请求隔离的数值验收仍待Lesson 4，不因长度重置就标记通过。

## 9. 现在继续学什么

先运行观察器CPU测试：

```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_cache_inspection -v
```

再读报告，回答：为什么实际模型缓存只有约26 MiB，GPU上的allocated却约9 GB？
这里要列出模型权重、模型缓存和诊断/临时开销，不把reserved当成有效缓存。
