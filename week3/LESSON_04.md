# Lesson 4：增量计算真的正确吗？

状态：已实现对拍工具并运行真实校准与诊断；候选容差审阅未通过，正式数值验收仍未完成。见[容差审阅](LESSON_04_CALIBRATION_REVIEW.md)与[实验报告](LESSON_04_RESULTS.md)。
前置：[Lesson 3 总结](LESSON_03_SUMMARY.md)。本课继续用 Qwen3.5-4B 的 Transformers 参考路径，不代表 nano-vLLM 原生适配已完成。

## 1. 从“状态更新了”到“更新正确”

Lesson 3 看到 KV 长度增长，Linear Attention 状态形状不变但数值更新。
错误实现也可能满足这些现象，所以接下来要比较输出数值。
比较对象是最后位置的完整 logits 向量，形状为 [1,V]，不是只看生成文本。

两条路径使用同一模型权重、token IDs、位置语义、mask 语义、精度与后端设置：

- 参考路径：每次从空状态重新处理整个前缀，use_cache=False。
- 缓存路径：一次 Prefill，然后每次只处理新 token，传入此前的混合缓存。

这是同一参考模型内的缓存一致性验证，不是两个独立引擎的正确性证明。

## 2. 为什么必须固定后续 token？

如果两条路径分别 argmax，某一步很小的误差可能让它们选出不同 token。
之后的输入前缀就不同了，后续 logits 差异混合了计算误差和输入差异，难以定位。

本课使用固定续接序列（teacher forcing）：事先给定相同 token，每次都将它送入两条路径，不用各自的预测决定下一次输入。
这里不训练、不反向传播；只是固定比较条件。

教学例子：prompt=[11,22,33]，固定 continuation=[44,55]。
这些数字仅用于手推；真实实验使用当前 tokenizer 编码并保存的合法文本 IDs。

| 比较点 | 全重算路径本次输入 | 缓存路径本次输入 | 缓存路径处理后长度 |
| --- | --- | --- | --- |
| Prefill | [11,22,33] | [11,22,33]，从空状态开始 | 3 |
| Decode 1 | [11,22,33,44] | [44]，带历史状态 | 4 |
| Decode 2 | [11,22,33,44,55] | [55]，带历史状态 | 5 |

每一行比较的是相同完整前缀之后的“下一个 token 的得分”。Prefill 也比较，避免只检查 Decode。
固定续接 2 个 token 会产生 3 个比较点；与自由生成 max_new_tokens=2 时只有一次 Decode 不同。

## 3. 对照现有 forward 接口

下面是核心调用的教学摘录，完整实现见 [verify_generation.py](verify_generation.py)：

```python
reference, _ = forward_last(
    model, prefix,
    processed_tokens=0, cache=None, use_cache=False,
)
candidate, cache = forward_last(
    model, token,
    processed_tokens=processed, cache=cache, use_cache=True,
)
```

第二次调用之前，cache 必须来自缓存路径自己的 Prefill 和之前的 Decode。
prefix 是包含当前 token 的完整前缀；processed 是缓存路径本次输入之前的已处理长度。
全重算每次从位置 0 开始，缓存 Decode 使用当前 token 的实际位置，不能每次重置为 0。
两条路径不得共享同一可变 cache。

## 4. 看哪些误差？

先验证 shape 相同且全部有限，再把 logits 转成 FP32 计算差异：

```python
diff = candidate.float() - reference.float()
max_abs = diff.abs().max()
mean_abs = diff.abs().mean()
```

- max_abs：最差的那个词表位置相差多少。
- mean_abs：全部词表位置平均相差多少。
- relative_l2：误差向量的 L2 范数除以参考向量的 L2 范数；参考全零时需单独处理。
- top1 与 top1/top2 margin：最高分 token 是否变化，前两名的差距是否很小。

top1 相同不代表所有 logits 正确；top1 不同也不能只凭“误差小”自动判通过。
没有经过独立校准、审阅并冻结的容差时，只能输出 diagnostic，不能声称 pass。
不要为了让待验收样本通过而临时放宽阈值。

## 5. 实验顺序与当前边界

1. 先用 CPU 假模型测试：正确缓存应与全重算一致；故意污染历史时检查器必须能报出差异。
2. 实现固定前缀逐步比较，记录 IDs、位置、模型和环境指纹。
3. 独立校准数值容差，审阅后冻结；再对真实验收文本运行。
4. 比较 B 单独运行、A 正常结束后 B、A 有状态执行后抛异常再运行 B。
5. 保留每一步证据，区分 diagnostic、pass、fail、needs_review。

完整校准方法遵循[已批准实现计划](../archive/development/superpowers/plans/2026-09-25-qwen35-text-baseline.md)的 Task 7。
已新增验证脚本与 CPU 检查器测试，已加载真实 BF16/FP32 模型完成独立校准。候选阈值过宽，不批准；真实验收文本仅作为 diagnostic 运行，不能将命令退出码 0 等同于正确性通过。请求隔离追加了 Prefill 成功后 Decode 异常与同一生成入口 B 的检查，实际数据见实验报告。

## 6. 先回答这一题

prompt 有 6 个 token，固定续接 token 为 A、B。完成 Prefill 比较后，继续比较两步。

1. 第一轮 Decode，两条路径分别输入哪些 token？缓存路径的 processed_tokens 参数是多少？
2. 第二轮 Decode，Full Attention 的 KV 长度应是多少？这一行 logits 预测的是 B 还是 B 后面的 token？
3. 如果第一轮两条路径 argmax 不同，下一轮还能分别喂各自选出的 token 来做严格对拍吗？

答完再手写固定前缀循环，不提前跳到性能优化。
