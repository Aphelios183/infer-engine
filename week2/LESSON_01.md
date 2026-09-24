# 第二周第一课：只算新 token，复用历史 K/V

第一周你已经验证了 MHA，也学会了先测量再解释瓶颈。本周我们先改变“重复做什么”，再讨论速度。

## 本周路线

| 课次 | 问题 | 产出 |
|---|---|---|
| 1（今天） | 什么可以缓存？Decode 的形状和 mask 怎样变化？ | 运行逐步演示，解释每一步，与完整前缀对拍 |
| 2 | 缓存如何追加、重置、检查容量？ | 亲手重写 append，完成生命周期与边界验证 |
| 3 | 缓存解码与完整重算到底省了什么？ | 设计公平性能对照，区分 Prefill 与 Decode |
| 4 | 显存怎样增长，预分配有什么代价？ | 计算有效/预留 KV 字节，完成总结 |

当前目录提供第 1–2 课的参考实现与验证，以及第三课讲义 LESSON_03.md。当前进度见 PLAN.md；第三课脚本和实测报告已完成，结果见 LESSON_03_RESULTS.md。

## 1. 为什么历史 K/V 可以复用

在当前单层 MHA 中，每个 token 的投影分别是 Q_i=x_i W_q、K_i=x_i W_k、V_i=x_i W_v。
固定权重、固定历史输入，增加一个新 token 不会改变旧 token 的 K/V。

若输入从 `[x1,x2,x3]` 变为 `[x1,x2,x3,x4]`：

- 完整重算：重新投影四个 token，计算四个位置的输出，再取最后一个。
- 缓存路径：复用 K1..K3 和 V1..V3，只投影 x4，计算最后一个位置的输出。

第 4 个 query 仍然需要与 K1..K4 打分，再加权 V1..V4。因此缓存并没有消除历史读取，也没有让每步 Attention 变成与历史长度无关。

完整的多层 causal Transformer 在确定性的推理设置下也能逐层缓存历史 K/V：历史位置不读取未来位置。真实模型通常每层有自己的缓存，并要处理位置编码。当前演示只有一层 MHA、随机 hidden states，没有 tokenizer、RoPE、采样或真实文字生成。

## 2. Prefill 与 Decode

Prefill：空缓存，一次处理提示词，例如 x1,x2,x3，写入 3 组 K/V。
Decode：接收新增输入，常见一次一个 token；保存新 K/V，用新 Q 读取整个有效缓存。

注意“输入的 token”和“预测的 token”不是同一个位置：完整语言模型在处理 x1..x3 后，可用最后位置 logits 选出 x4；随后处理 x4，才把 x4 的 K/V 写入，并预测 x5。不要在 token 尚未经过模型时就说它已经被缓存。

本课使用 B=1、C=8、H=2，所以 Dh=4：

| 阶段 | 新输入 | Q_new | K_all / V_all | scores | 输出 |
|---|---|---|---|---|---|
| Prefill 3 个 | [1,3,8] | [1,2,3,4] | [1,2,3,4] | [1,2,3,3] | [1,3,8] |
| Decode 第 4 个 | [1,1,8] | [1,2,1,4] | [1,2,4,4] | [1,2,1,4] | [1,1,8] |
| Decode 第 5 个 | [1,1,8] | [1,2,1,4] | [1,2,5,4] | [1,2,1,5] | [1,1,8] |

Q 是新位置的，K/V 是所有有效位置的；`new_tokens` 与 `cache.length` 要分清。

## 3. Decode 的 causal mask 按绝对位置构造

缓存已有 3 个 token，新 query 是绝对位置 3（从 0 开始）。它能看 key 位置 0、1、2、3：

```text
             K0 K1 K2 K3
Q3 可看：     1  1  1  1
Q3 屏蔽：     0  0  0  0
```

直接对 `[1,4]` 矩阵取上三角，会误把这个 query 当作位置 0，只允许看第一个 key。
本课 `causal_mask` 使用 `key_position > query_position` 标记屏蔽位置。
若一次追加两个新 token，前一个新 query 还不能看后一个新 key；因此不能把所有 chunk 的 mask 都去掉。

## 4. Cache 的 length 和 capacity

`KVCache` 预分配 `[B,H,max_seq_len,Dh]` 两块张量。
`capacity` 表示预留多少位置；`length` 表示实际写入多少位置。
`view()` 只返回前 length 个位置，不能让 Attention 读到 torch.empty 的未初始化区域。

追加时往 `[length:length+new_tokens]` 写入，然后更新 length。这里复用存储，没有每步把完整旧缓存用 torch.cat 复制到新张量。
新请求必须 reset，且不能在同一缓存里混用不同模型权重或不同批次顺序。

本层 KV 字节数为 `2 × B × H × 有效长度 × Dh × 每元素字节数`。预留字节把有效长度换成 capacity。
标准未压缩的多层 KV 再按各层相加；GQA/MQA 应使用 KV 头数，不能总拿 Query 头数代入。
这些只是 KV 张量字节，不等于整个进程 GPU 显存：还包括权重、激活、临时张量和框架开销。

## 5. 今天怎么运行

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
python week2/step_by_step.py
python week2/verify_kv_cache.py
```

这两条默认跑 CPU，足够验证概念。如果想在 GPU 对拍，先确认 GPU 2 空闲：

```bash
CUDA_VISIBLE_DEVICES=2 python week2/verify_kv_cache.py --device cuda
```

验证覆盖 T=1、多 batch、逐 token 与多 token chunk 对拍、矩形 mask、容量/形状/dtype 错误、reset 后隔离和存储复用。
它验证的是当前教学 MHA 的机制，不代表已经实现了完整语言模型或证明了性能收益。

## 6. 今天的阅读顺序

1. 先手算形状表，再运行 step_by_step.py 核对。
2. 读 kv_cache.py 中 forward_cached 的三步：投影新 token、追加 K/V、新 Q 读取全部 K/V。
3. 再读 append/view/reset，区分缓存的内容与管理信息。
4. 完成 EXERCISES.md 的第一课题目，先解释，再写优化。
