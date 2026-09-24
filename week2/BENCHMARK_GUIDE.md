# Lesson 3 实验操作说明

状态：脚本已准备；按你的要求，暂不运行正式性能实验，也不使用 GPU。小规模 CPU 单元测试只验证代码流程，不是性能数据。

## 1. 本次新增文件

- benchmark_kv_cache.py：比较完整前缀重算与 KV Cache，默认只打印计划。
- test_benchmark_kv_cache.py：验证工作负载、逐步对拍、重置、预热轮次隔离与统计。
- 本说明：何时运行、怎样复现、如何解释数据。

参考计算路径没有改动。脚本使用 kv_cache.py 的参考类，不会自动使用 kv_cache_exercise.py 的独立手写函数。

## 2. 现在可以做：查看计划，不跑实验

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
python week2/benchmark_kv_cache.py --device cuda
```

没有 --run，只打印 planned_only 与参数，不分配模型/缓存、不执行模型、不计时、不写结果文件，也不需要可用 GPU。不要把屏幕上的配置当作已完成实验。

默认参数：B=2、C=64、H=4、FP32、N=16；L=32/128/512/1024；3 轮预热、10 轮正式测量；种子 42；CPU 线程数 1；正式运行时关闭 TF32。

## 3. 以后获准运行时：先小配置，再完整配置

先运行 nvidia-smi，确认某张卡可供实验使用；低瞬时利用率或剩余显存不代表独占。不要终止其他人的进程，也不要沿用“GPU 2 以前空闲”的假设。

以下命令供以后使用，不是当前已执行的命令。GPU_ID=2 只是编号示例，必须改成届时确认可用的编号。

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
nvidia-smi
GPU_ID=2

# 首次只跑短序列，确认输出和数据格式。
CUDA_VISIBLE_DEVICES=$GPU_ID python week2/benchmark_kv_cache.py \
  --device cuda --lengths 32 128 --run

# GPU 可独占且前一步正常后，单独运行完整配置。
CUDA_VISIBLE_DEVICES=$GPU_ID python week2/benchmark_kv_cache.py \
  --device cuda --lengths 32 128 512 1024 \
  --steps 16 --warmup 3 --repeats 10 --run
```

每次默认创建带 UTC 时间戳的新文件：week2/results/kv_cache_*.json。也可用 --output 指定路径；若文件已存在会拒绝覆盖。CUDA 不可用会报错，不会悄悄改跑 CPU。

CPU 流程检查命令（以后需要时才运行）：

```bash
python week2/benchmark_kv_cache.py --device cpu --lengths 8 \
  --steps 2 --warmup 1 --repeats 2 --run
```

CPU 耗时不是 L40 数据，不能和 GPU 结果混成一个表。

## 4. 一轮实验的精确边界

创建模型、准备输入、分配容量 L+N 的缓存都在计时外。先检查 Prefill 和每一个 Decode 输出与完整重算一致；不通过则终止，不输出加速结论。

完整重算：计时一次 Prefill，再单独计时 N 次长度 L+1 到 L+N 的完整前缀重算。

缓存路径：reset 在计时外；计时一次 Prefill，包含初始 KV 写入；再单独计时 N 次单 token 输入，包含每步的新 Q/K/V 投影、append、Attention 和输出投影。

GPU 每个阶段的开始之前和结束之后同步；N 步 Decode 内部不逐步同步。每轮重新 Prefill，不能沿用预热或上一轮的状态。正式轮次交替两条路径的先后顺序。

total_ms 是一轮内两个独立同步阶段耗时之和。它不包含两阶段之间的 Python 管理间隔、初始化、分配、reset、网络等，不是额外测量的一次连续端到端请求延迟。

## 5. JSON 怎样读

- config：工作负载和测量参数。
- metadata：环境、实际 GPU 名称、CUDA_VISIBLE_DEVICES、TF32、源码 SHA256 和命令参数。
- results[].correctness：逐步对拍次数、最大绝对误差、容差。
- results[].raw.baseline/cache：每轮 Prefill、Decode、平均每步和两阶段合计时间，单位都是 ms。
- results[].orders：每轮执行顺序。
- results[].summary：各指标中位数、最小值和最大值。
- results[].speedup_ratio_of_medians：完整重算中位耗时 / 缓存中位耗时；分别报告 Prefill、Decode 和合计。
- kv_used_bytes_at_end / kv_allocated_bytes：本层 K/V 张量字节，不是进程显存占用。

合计中位数来自“各轮先相加，再取中位数”，不是“两个阶段的中位数相加”。GPU 前后快照只能辅助记录，不能证明整个实验期间没有竞争。

## 6. 跑完以后你需要回答

1. Prefill 哪边更快？是否有额外缓存写入成本？先描述结果，不预设方向。
2. Decode 加速比和合计加速比是否相同？为什么不能互相替代？
3. L 增长时，两条路径如何变化？哪些解释是实测支持的，哪些只是推测？
4. 短序列是否明显加速？若没有，下一步如何用单独的 profiler 运行验证提交开销假设？

正式数据出来后再创建 LESSON_03_RESULTS.md，保留环境、命令、原始 JSON、数据表及限制。不用单层教学实验声称优于生产推理引擎，不用 profiler 开启时的耗时填正式表。
