# Week 1：用证据判断 Attention 的固定开销

你已验证 MHA 的输出正确。本实验接着回答：T=1、16、64、256 的时间相近，到底是谁在花时间？

目标是建立「提出假设 → 做对照 → 看时间线 → 限定结论」的习惯。最终不一定得到“全是 launch 开销”的答案。

## 1. 先预测，再运行

写下你对下面四个版本的预测，并写一句原因：

| 版本 | 改变了什么 | 保留了什么 |
|---|---|---|
| eager | 原始 `mha.py` | 全部原始计算 |
| cached_mask | mask 在计时前构造一次 | QKV、scores、masked_fill、softmax、AV、输出投影 |
| sdpa | Attention 交给 PyTorch SDPA | 同样的权重、输入、输出语义、QKV/输出投影 |
| cuda_graph | 把原始 MHA 捕获后 replay | 原计算图，包括每轮 mask 构造的 GPU 操作 |

这些版本都先与原始 MHA 做 FP32 对拍（atol=1e-5, rtol=1e-4），才进入性能测量。
CUDA Graph 用固定形状、固定输入地址；捕获和预热不计入稳态耗时。这是控制实验，不是通用模型服务。
本实验使用方形 causal self-attention；不要直接把此处的 mask 对照方法套到带历史缓存的矩形 Decode。

## 2. 运行命令

在远程服务器执行；先用 `nvidia-smi` 确认 GPU 2 仍空闲。`CUDA_VISIBLE_DEVICES=2` 后程序内 `cuda:0` 对应物理 GPU 2。

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
nvidia-smi

# 第一次只看 T=1 和 64，缩短等待时间。
CUDA_VISIBLE_DEVICES=2 python week1/profile_attention.py \
  --seq-lens 1 64 --profile-seq-lens 1 64 --iters 50 --repeats 3

# 完整实验：与原 verify.py 相同的 B=8、C=64、H=4、FP32。
CUDA_VISIBLE_DEVICES=2 python week1/profile_attention.py

# 不采集 trace，仅复测稳态耗时，检查结果是否稳定。
CUDA_VISIBLE_DEVICES=2 python week1/profile_attention.py --skip-profiler

# 可选对照：固定为 1 个 CPU intra-op 线程，其他参数保持不变。
CUDA_VISIBLE_DEVICES=2 python week1/profile_attention.py --threads 1 --skip-profiler
```

每次生成新的 `results/week1/<UTC时间>/`，不会覆盖旧实验。`results/` 已被仓库忽略。
脚本保留 PyTorch 默认 CPU 线程数，除非显式传入 `--threads`。记录了线程数、TF32 设置、源码 SHA256、git 状态、GPU 前后快照等信息。
默认采用 10 次预热、100 次迭代、5 轮重复。首轮尝试建议使用默认形状，避免大 attention 矩阵造成不必要的内存占用。

输出文件：

- `report.md`：先看四组未经 profiler 干扰的计时，再看采集期的诊断统计。
- `summary.json`：原始重复样本、数值误差、计时口径、环境、逐次同步对照。
- `T64_eager.operators.txt`：按 CPU self time 排序的算子表，以及单独按实际 GPU kernel 总时长排序的表。
- `T64_eager.trace.json`：CPU、CUDA API、GPU kernel 的执行时间线。
- 其他 T 和版本对应同名格式文件。

`--output` 可以指定一个新的目录；如果目录已存在会拒绝覆盖。

## 3. 四个计时字段分别在测什么

GPU 通常异步执行：CPU 提交工作后可以继续走，GPU 在另一条时间线上处理。

```text
CPU：  准备/提交 A → 准备/提交 B → ... → 提交完毕 → 等待最后任务结束
GPU：       执行 A → 可能空档 → 执行 B → ... → 最后任务结束
```

- **Wall**：整批 forward 从提交到 GPU 完成的墙钟时间，除以次数。是流水执行的平均成本，不是独立请求的延迟。
- **CPU 提交**：Python 循环和框架调用返回所需的时间。包括 Python、dispatcher、CUDA runtime，且可能包含内部阻塞；不是纯 CPU 计算时间，更不是某一个 launch API 的时间。
- **GPU stream 区间**：前后两个 CUDA Event 之间的时间。可能包含 GPU 等待后续任务的空档，不能直接叫 kernel 时间。
- **尾部等待**：最后一个 forward 返回后，记录结束事件并等它完成的时间，按次数平均。它只是最后剩下的等待，并非全部 GPU 耗时。

CPU 与 GPU 大量重叠：**不把 CPU 提交与 stream 区间相加；也不把 Wall 减去 stream 区间当成 CPU 开销。**

`sync_each` 是另一项控制实验：每次 forward 都同步。它包含频繁同步的成本，通常不能与批量异步循环视为同一负载。
原始 `verify.py` 的计时是整个循环结束后才同步，更接近本实验的 Wall。

## 4. 从算子表开始找线索

打开 `T64_eager.operators.txt`：

1. 先找 `aten::linear`、`aten::mm/bmm`、`aten::softmax`、`aten::triu`、`aten::masked_fill`、`aten::clone/copy_`。
2. 看 calls：一次 MHA 拆成了多少次小操作？哪些操作没有大矩阵计算，却每轮都重复？
3. 分开看 CPU self time 与 GPU kernel 表。`total` 可包含子算子时间，父子行相加会重复统计。GPU 表只统计 trace 中的实际 kernel，不把覆盖多个 kernel 的阶段标记计为额外计算。
4. 不要看到 `cudaDeviceSynchronize` 占 CPU 时间多就认定“同步函数计算很慢”：它往往在等之前排入队列的 GPU 工作完成。

profiler 中加入了八个中文教程对应的英文阶段标记，从 `01_QKV_projection_and_split` 到 `08_output_projection`。
这些标记只在采集路径使用；额外标记、shape 记录、CUPTI 采集会扰动微秒级执行，不能拿采集时的耗时当最终加速比。

## 5. 打开 trace，看 CPU 与 GPU 的先后关系

把目标 `.trace.json` 从服务器下载到本地，在浏览器打开 https://ui.perfetto.dev/，选择 Open trace file。
选择本地文件进行查看即可；无需发布或创建分享链接。不要同时打开所有 trace，先看 `T64_eager`。

按以下顺序观察：

1. 搜索 `forward_0`，找到 CPU 侧的八个阶段。缩放到一个或几个 forward 的尺度。
2. 找 GPU stream 的 kernel 行；不要把 CPU 的 `aten::*` 矩形误认成 GPU kernel。
3. 看 kernel 之间是否有可见空档；结合 CUDA runtime 调用和关联箭头，找相应的 CPU 发起操作。GPU 事件可能落在 CPU forward 范围之外，这是异步执行的结果。
4. 打开 `T64_cuda_graph`：提交方式是否简化？在 GPU trace 采集完整的前提下，核心 kernel 名称是否保留？总数可能增加，因为 graph replay 也会产生辅助操作；不要预设 graph 一定减少 kernel 数量。
5. 打开 `T64_cached_mask`：检查 mask 构造相关操作是否消失，`masked_fill` 是否仍然存在。
6. 打开 `T1024_eager`：大矩阵 kernel 是否占据更长的连续时间？再与 T=64 对照。

`summary.json` 自动记录每轮 kernel 数、kernel 时长之和，以及 trace 内首个到最后一个 GPU 活动之间的空档。
空档用 kernel/memcpy/memset 区间的并集计算，避免多 stream 重叠被重复累计。
这些是**采集期**统计，不是 nvidia-smi 的 GPU 利用率。若没有 GPU kernel 事件，会记录 unavailable/null；不能把缺失数据解释成“GPU 耗时为零”。

## 6. 怎样把线索变成结论

| 观察 | 可以支持的解释 | 不能直接断言 |
|---|---|---|
| 短序列 Wall 变化小 | 存在不随序列长度明显增长的成本 | 一定全是 launch |
| 缓存 mask 有收益 | mask 构造及相关分配/调度值得关注 | Attention 算法变快了同样比例 |
| Graph 明显加速，计算语义不变 | 主机提交、调度或分配路径具有可优化成本 | 收益全等于 cudaLaunchKernel 时间 |
| SDPA 加速 | 替换 Attention 执行路径有收益 | 必然调用了 FlashAttention |
| GPU trace 有空档 | 需要检查供给不足、同步或设备干扰 | 仅凭空档证明 CPU 是唯一瓶颈 |
| T=1024 比 T=64 慢很多 | 数据规模相关的开销开始明显 | 已证明是算力瓶颈或带宽瓶颈 |

本 MHA 显式物化 `[B,H,T,T]` 的 scores/weights。B=8、H=4、T=1024、FP32 时，单个这样的张量约 128 MiB；T=64 时约 0.5 MiB。
因此长序列不仅有更多计算，也有更多内存读写和临时张量。要区分算力与带宽瓶颈，需要进一步的硬件计数器或专门控制实验，不能只看本实验的耗时。

CUDA Graph 也改变了内存分配复用方式。SDPA 默认自动选择后端；以实际算子/kernel 名称为准。
实验记录 GPU 前后负载，但这不等于保证全过程独占。若重复样本波动明显，先换空闲时段复测。

## 7. 你的验收作业

用本次数据填写，先写证据再写结论：

1. T=64 的 eager Wall 为 ___ ms，Graph 为 ___ ms，加速 ___ 倍；原始重复样本波动为 ___。
2. eager 与 Graph 的 kernel 数分别为 ___ / ___；如果采集不完整，写“无法比较”。
3. cached_mask 是否更快？消失了哪些操作？如果差异落在波动范围内，写“暂不能确认收益”。
4. T=1024 最耗 GPU 时间的三种 kernel 是什么？对应代码的哪几步？
5. CPU self time 最大的操作是否对应 GPU kernel 时间最多的计算？为什么？
6. 用三句话写“观察到什么、支持什么、还不能证明什么”。

能解释上述六项，就完成 Week 1 的性能分析部分，可以进入 KV Cache。

参考：
- PyTorch CUDA 异步执行与计时：https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch Profiler 教程：https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- PyTorch CUDA Graph：https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
