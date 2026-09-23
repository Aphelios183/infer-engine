# 第二周第二课：亲手管理 KV Cache 的生命周期

本课接在 LESSON_01.md 后面。目标不是增加优化技巧，而是把缓存写对：知道数据写到哪里、哪些数据能读、什么时候必须重置，以及错误输入为什么不能破坏已有状态。

预计用时 60–90 分钟。只需要现有 nanovllm 环境和 CPU，不需要占用 GPU。本课对应 kv_cache.py 中的 KVCache；不修改参考实现，也不覆盖你已填写的 EXERCISES.md。

## 1. 先纠正上一轮练习的一个细节

条件：B=2、H=4、Dh=8，历史长度为 6，本次输入 2 个新位置。

| 张量 | 形状 | 含义 |
|---|---|---|
| Q_new | [2,4,2,8] | 两个新增位置的 query |
| K_new / V_new | [2,4,2,8] | 本次需要写入的 K/V |
| K_all / V_all | [2,4,8,8] | 6 个旧位置加 2 个新位置 |
| scores | [2,4,2,8] | 两个 query 各对八个 key 打分 |
| 合并多头后的输出 | [2,2,32] | 只返回新增位置的输出 |

你对可见范围的理解是正确的：第七个位置只能看前七个，第八个能看全部八个。需要注意总长度是 6+2=8，不能沿用单 token 追加时的 7。

```text
             K1 K2 K3 K4 K5 K6 K7 K8
第7个 query   0  0  0  0  0  0  0  1
第8个 query   0  0  0  0  0  0  0  0
```

这里 1 表示屏蔽，0 表示允许。先通过 QKᵀ/√Dh 得到 scores，把被屏蔽的分数设为 -inf，再沿最后一维做 softmax。softmax 负责把得分转换为权重，不负责计算 QK 点积。

## 2. 缓存是一块存储，加一个有效长度

```python
self.k = torch.empty(B, H, capacity, Dh)
self.v = torch.empty_like(self.k)
self.length = 0
```

- capacity：预留了多少位置，等于这里张量第三个维度的大小。
- length：当前请求已经写入多少位置。
- [0:length)：可以参与 Attention 的有效区域。
- [length:capacity)：未使用区域，不允许直接参与 Attention。

torch.empty 不会保证内容为零。即使把这块空间初始化成零，也不能直接让未使用位置参加 Attention：它们仍会进入 softmax 的归一化，改变有效位置的权重。

本课始终维护三个约束：

1. 0 <= length <= capacity。
2. 前 length 个位置的 K 和 V 都已经写好，并属于当前请求。
3. Attention 通过 view() 读取有效区域，不直接读取整块预分配张量。

## 3. append：先检查，再写入，最后更新长度

已有 6 个位置，本次追加 2 个，容量是 16：

```text
Python 下标   0 1 2 3 4 5 | 6 7 | 8 ... 15
追加之前      有效旧缓存   |     空闲区域
追加之后      保持不变     | 新KV | 剩余空闲
length        6 → 8
capacity      始终为 16
```

核心操作如下。这是讲解片段；完整检查逻辑请看 kv_cache.py 的 append。

```python
count = k_new.size(2)
start = self.length
end = start + count
if end > self.capacity:
    raise ValueError("缓存容量不足")
self.k[:, :, start:end, :].copy_(k_new)
self.v[:, :, start:end, :].copy_(v_new)
self.length = end
```

为什么不能先修改 length 再写？因为 length 是“已经写好”的承诺，而不是“打算写入”的数量。

写入前还要检查：K/V 同为四维且形状相同；B、H、Dh 与缓存匹配；新增长度大于零；device 和 dtype 匹配。对这些可预先发现的输入错误，应在任何写入发生前抛出异常，使原来的 length 和数据保持不变。这不等于承诺任意硬件故障下都具有事务回滚能力。

为什么不每步用 torch.cat？拼接通常需要新存储并复制旧内容。预分配让每次只写新 K/V，避免这部分反复复制；但它并不消除 Attention 对历史 K/V 的读取，也不保证所有小规模场景都更快。性能留到第三课实测。

## 4. view：给计算层一个正确的数据边界

```python
def view(self):
    return self.k[:, :, :self.length, :], self.v[:, :, :self.length, :]
```

本项目的 view() 是自定义方法名，不是在调用 Tensor.view 来 reshape。它通过基本切片返回共享底层存储的张量，没有复制整份 K/V。

例如底层 k 的形状为 [2,4,16,8]，length=8，返回的 k_all 形状就是 [2,4,8,8]。

两个注意点：

- 切片不一定 contiguous，不要把“返回视图”理解成“必然连续”。
- 返回值不是历史快照。后续 reset 并重写同一片存储时，以前保存的视图也可能观察到变化；需要独立快照时才用 clone()。

## 5. reset：逻辑清空，不等于物理擦除

```python
def reset(self):
    self.length = 0
```

请求 A 结束，独立请求 B 开始：

1. reset 后 length=0，view() 返回长度为 0 的有效区域。
2. B 的新 K/V 从下标 0 开始覆盖写入。
3. 只在 K/V 都写好后更新 length。
4. B 的 Attention 只读取前 length 个位置。

因此，在这个串行教学实现中，严格遵守上述流程，B 不会通过有效缓存读到 A 的历史内容；没有必要为计算正确性每次清零整块张量。

但旧数据仍可能存在于未覆盖的底层区域。reset 不是安全擦除，也不释放预分配内存。不要把本课的逻辑隔离等同于多租户安全隔离；有旧视图、并发读取或跨 CUDA stream 使用时，还需要额外的生命周期和同步管理，本课没有实现这些机制。

同一请求继续生成时不要 reset；换独立请求、模型权重或不兼容的批次对应关系时不能盲目沿用旧缓存。本实现只有一个 length，假设 batch 中所有样本等长推进，不能直接支持各请求独立增长和退出。

## 6. 动手实验：旧内容还在，新请求却读不到

先预测输出，再把下面整段复制到服务器终端执行。这里用 11 和 22 作为容易辨认的 K/V 标记，不是在运行语言模型。

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
python - <<'PY'
import torch
from week2.kv_cache import KVCache

cache = KVCache(1, 1, 8, 1)
storage_before = cache.k.data_ptr()

# 请求 A 使用前三个槽位。
a = torch.full((1, 1, 3, 1), 11.0)
cache.append(a, a)
print("A 有效内容:", cache.view()[0].flatten().tolist())
assert cache.length == 3

# 重置有效边界，不擦除底层数据。
cache.reset()
print("reset 后有效形状:", tuple(cache.view()[0].shape))
assert cache.length == 0
assert cache.view()[0].numel() == 0

# 请求 B 只写两个槽位。
b = torch.full((1, 1, 2, 1), 22.0)
cache.append(b, b)
print("B 有效内容:", cache.view()[0].flatten().tolist())
# 仅为观察残留而直接读取已知被 A 写过的第三槽；Attention 不应这样读。
print("底层第三槽仍是:", cache.k[0, 0, 2, 0].item())
print("底层存储地址不变:", storage_before == cache.k.data_ptr())
assert cache.view()[0].flatten().tolist() == [22.0, 22.0]
assert cache.k[0, 0, 2, 0].item() == 11.0
assert storage_before == cache.k.data_ptr()

# 2+7 超过容量 8：拒绝写入，旧有效数据不能受影响。
before_k, before_v = (t.clone() for t in cache.view())
too_many = torch.zeros(1, 1, 7, 1)
try:
    cache.append(too_many, too_many)
except ValueError as exc:
    print("预期的越界错误:", exc)
else:
    raise AssertionError("应该拒绝超容量写入")
assert cache.length == 2
torch.testing.assert_close(cache.view()[0], before_k)
torch.testing.assert_close(cache.view()[1], before_v)
print("生命周期实验通过")
PY
```

你应观察到：A 有效内容是三个 11；reset 后形状为 [1,1,0,1]；B 有效内容是两个 22；底层第三槽仍为 11；底层存储地址没变；超容量追加被拒绝。

## 7. 亲手实现，不改参考答案

在 /home/ubuntu/infer-engine/week2/kv_cache_exercise.py 中，自己写一个同接口的 KVCache 类。先不要复制参考实现：

1. __init__：分配 K/V，初始化 length，提供 capacity。
2. append：完成形状、dtype、device、容量检查以及原位追加。
3. view：只返回有效区域。
4. reset：保留存储，重置有效长度。

先完成核心四步，再补 used_bytes 和 allocated_bytes 属性，与参考类对齐。

建议的自测顺序：空缓存 → 追加 3 个 → 再追加 2 个 → 确认前 3 个不变 → 尝试越界并确认状态不变 → reset → 写入另一个请求。

将上节实验的 import 改为 from week2.kv_cache_exercise import KVCache，可以检查你自己的核心实现。这个实验只是部分检查，还应补 K/V 形状不一致、dtype 不一致、新增长度为零，以及旧 V 不被覆盖的用例。

现有参考实现的完整验证命令是：

```bash
cd /home/ubuntu/infer-engine
python week2/verify_kv_cache.py
```

注意：这条命令默认测试 kv_cache.py，不会自动测试你新建的练习类。要复用其测试，可另存测试副本，并在副本中仅替换 KVCache 的导入；causal_mask、forward_cached 等仍从参考模块导入，保留原测试文件不动。

## 8. 本课验收与下一课

请用自己的话回答：

- 为什么清零未使用槽位，也不能让它们直接参加 softmax？
- 为什么先做输入检查，再写 K/V，最后更新 length？
- reset 后显存为什么没有随 length 一起变成零？
- B 为什么不会通过 view() 读到 A 的数据？哪些使用方式会破坏这个前提？
- 两个 batch 样本分别已有 3、7 个位置，一个共享 length 为什么不够？

验收标准：能独立写出 append/view/reset，能解释生命周期实验，能让错误输入不破坏原有有效缓存。完成后进入第三课：分别测量 Prefill 和 Decode，对比缓存与完整前缀重算，分析省下的计算和新增的存储成本。今天无需提前做分页、多 GPU 或 CUDA kernel。
