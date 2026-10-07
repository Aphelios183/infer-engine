<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

一个从零实现的轻量级 vLLM 推理引擎。

## 核心特性

* **快速离线推理** - 推理速度与 vLLM 相当
* **代码可读性强** - 核心库仅约 1200 行 Python 代码，简洁清晰
* **完整优化套件** - 前缀缓存、张量并行、Torch 编译、CUDA Graph 等

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

**依赖项：**
- Python >= 3.10, < 3.13
- torch >= 2.4.0
- triton >= 3.0.0
- transformers >= 4.51.0
- flash-attn
- xxhash

## 模型下载

手动下载模型权重：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

```python
from nanovllm import LLM, SamplingParams

# 初始化引擎（enforce_eager=True 禁用 CUDA Graph，tensor_parallel_size 指定并行数）
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)

# 设置采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 批量生成
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

详细用法请参考 `example.py`。API 与 vLLM 接口基本一致，`LLM.generate` 方法有少许差异。

## 性能测试

测试脚本见 `bench.py`。

**测试环境：**
- 硬件：RTX 4070 Laptop（8GB 显存）
- 模型：Qwen3-0.6B
- 请求总数：256 条序列
- 输入长度：100-1024 token 随机采样
- 输出长度：100-1024 token 随机采样

**性能对比：**

| 推理引擎 | 输出 Token 数 | 耗时 (s) | 吞吐量 (tokens/s) |
|---------|------------|---------|-------------------|
| vLLM    | 133,966    | 98.37   | 1361.84           |
| Nano-vLLM | 133,966  | 93.41   | **1434.13**       |

## 项目结构

```
nanovllm/
├── __init__.py                 # 包导出：LLM, SamplingParams
├── llm.py                      # LLM 类（LLMEngine 的薄封装）
├── config.py                   # 配置 dataclass
├── sampling_params.py          # 采样参数 dataclass
├── engine/
│   ├── llm_engine.py           # 核心引擎：调度循环、generate()
│   ├── model_runner.py         # GPU 模型执行、CUDA Graph、KV Cache 分配
│   ├── scheduler.py            # Prefill/Decode 调度、抢占策略
│   ├── block_manager.py        # PagedAttention 块管理 + 前缀缓存
│   └── sequence.py             # 序列状态跟踪
├── layers/
│   ├── attention.py            # FlashAttention + Triton KV Cache 写入核
│   ├── linear.py               # 张量并行线性层（Column/Row/QKV/Merged）
│   ├── embed_head.py           # 词表并行 Embedding + LM Head
│   ├── activation.py           # SiLU-and-Multiply 激活函数
│   ├── rotary_embedding.py     # 旋转位置编码（RoPE）
│   ├── layernorm.py            # RMSNorm（含融合 Add 变体）
│   └── sampler.py              # GPU 端随机采样
├── models/
│   └── qwen3.py                # Qwen3 模型完整实现
└── utils/
    ├── context.py              # 全局推理上下文
    └── loader.py               # Safetensors 权重加载器
```

## 核心设计

### PagedAttention + 前缀缓存

KV Cache 按固定大小的块（默认 256 token）管理。每个块通过 xxhash 进行内容哈希（链式哈希，与前序块关联）。当新序列与已计算序列共享前缀时，可复用相同的 KV Cache 块（引用计数管理），避免重复计算。

### 连续批处理（Continuous Batching）

调度器交错执行 Prefill 和 Decode：
1. **Prefill 阶段**：优先处理等待队列中的序列，支持分块 Prefill（长提示词可分多步调度）
2. **Decode 阶段**：调度所有运行中的序列。当显存不足时，抢占（驱逐）优先级最低的序列，将其块释放并重新排队

### CUDA Graph

Decode 阶步（batch size <= 512）使用预捕获的 CUDA Graph，减少内核启动开销。支持的 batch size：1, 2, 4, 8, 16, 32, ..., 512，共享内存池。

### Torch 编译

以下组件使用 `@torch.compile` 自动融合内核：
- RMSNorm（含融合 Add 变体）
- SiluAndMul 激活函数
- RotaryEmbedding 旋转位置编码
- Sampler 采样器

### 自定义 Triton 核

`store_kvcache_kernel`：高效的 Triton 核，通过 slot_mapping 将 K/V 张量写入分页 KV Cache。每个 Triton 程序实例处理一个 token 的所有 head 和维度数据。

### 张量并行（Tensor Parallelism）

完整的多卡并行支持（NCCL）：
- **列并行**：QKV 投影、Gate/Up 投影
- **行并行**：输出投影、Down 投影
- **词表并行**：Embedding 和 LM Head
- 进程间通信：共享内存 + pickle 序列化

### 高效序列序列化

`Sequence` 类自定义 `__getstate__`/`__setstate__`，Decode 阶段仅传输最后一个 token（而非完整历史），最小化进程间数据传输。

## 采样参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `temperature` | 1.0 | 采样温度（必须 > 1e-10，不支持纯贪心解码） |
| `max_tokens` | 64 | 最大生成 token 数 |
| `ignore_eos` | False | 是否忽略 EOS token |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_num_batched_tokens` | 16384 | 单批次最大 token 数 |
| `max_num_seqs` | 512 | 最大并发序列数 |
| `max_model_len` | 4096 | 最大序列长度（受模型配置限制） |
| `gpu_memory_utilization` | 0.9 | GPU 显存用于 KV Cache 的比例 |
| `tensor_parallel_size` | 1 | 张量并行度（最大 8） |
| `enforce_eager` | False | 是否禁用 CUDA Graph |
| `kvcache_block_size` | 256 | KV Cache 块大小（必须为 256 的倍数） |

## 支持的模型

目前仅支持 **Qwen3** 系列模型。扩展新模型只需在 `models/` 目录下创建对应的模型文件，遵循相同模式即可。

## 许可证

MIT License - Copyright (c) 2025 Xingkai Yu
