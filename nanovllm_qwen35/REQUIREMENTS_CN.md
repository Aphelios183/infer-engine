# Nano-vLLM 需求文档

## 1. 项目概述

### 1.1 项目背景

vLLM 是当前主流的大语言模型高吞吐推理引擎，但其代码量庞大、依赖复杂（Ray、自定义 CUDA 扩展、HTTP 服务框架等），不利于学习和二次开发。Nano-vLLM 旨在以最小代码量（约 1200 行）从零复现 vLLM 的核心推理能力，提供一个可读、可学习、可扩展的轻量级替代方案。

### 1.2 项目目标

| 目标 | 说明 |
|------|------|
| 功能对标 | 实现 vLLM 核心推理功能：PagedAttention、连续批处理、前缀缓存、张量并行 |
| 性能对标 | 推理吞吐量与 vLLM 在同一硬件上持平（差距 < 10%） |
| 代码精简 | 核心库控制在 1500 行以内，单文件不超过 300 行 |
| 依赖最小 | 仅依赖 torch、triton、transformers、flash-attn、xxhash |
| 易于扩展 | 新增模型只需在 `models/` 目录下添加一个文件 |

### 1.3 目标用户

- LLM 推理系统的学习者和研究者
- 需要轻量级离线推理引擎的开发者
- 希望在自定义场景中集成高效推理的工程师

---

## 2. 功能需求

### 2.1 核心推理引擎

#### 2.1.1 离线批量推理

- **FR-1.1**：支持接收多条文本提示词，批量生成回答
- **FR-1.2**：支持预分词的 token ID 列表作为输入
- **FR-1.3**：返回结构化输出，包含生成文本和 token ID 列表
- **FR-1.4**：支持 tqdm 进度条显示推理进度和吞吐量

#### 2.1.2 采样策略

- **FR-2.1**：支持 temperature 采样（温度范围 > 1e-10）
- **FR-2.2**：支持 top-p（nucleus）采样
- **FR-2.3**：支持 top-k 采样
- **FR-2.4**：支持 repetition penalty（重复惩罚）
- **FR-2.5**：支持 EOS token 检测和序列终止
- **FR-2.6**：支持 ignore_eos 模式（强制生成到 max_tokens）
- **FR-2.7**：GPU 端采样（Gumbel-max trick），避免 CPU-GPU 同步

#### 2.1.3 KV Cache 管理

- **FR-3.1**：PagedAttention 分页 KV Cache，按固定块大小（256 token）管理
- **FR-3.2**：根据 GPU 显存自动计算可用块数量
- **FR-3.3**：KV Cache 以连续张量分配，形状为 `(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`
- **FR-3.4**：每个注意力层通过视图引用对应的 Cache 切片

#### 2.1.4 前缀缓存（Prefix Caching）

- **FR-4.1**：基于 xxhash（xxh64）对每个块的 token 内容进行哈希
- **FR-4.2**：链式哈希，当前块哈希依赖前序块哈希，保证前缀唯一性
- **FR-4.3**：全局哈希表 `hash_to_block_id` 映射内容哈希到块 ID
- **FR-4.4**：新序列分配时自动检测前缀命中，复用已有块（引用计数递增）
- **FR-4.5**：块释放时引用计数递减，归零后回收至空闲池

#### 2.1.5 连续批处理（Continuous Batching）

- **FR-5.1**：调度器在每个 step 中决定执行 Prefill 还是 Decode
- **FR-5.2**：Prefill 优先于 Decode
- **FR-5.3**：支持分块 Prefill（Chunked Prefill）：长提示词可跨多个 step 分段调度（仅限批次首条序列）
- **FR-5.4**：序列完成后自动释放资源，新序列可立即加入批次
- **FR-5.5**：支持抢占（Preemption）：显存不足时驱逐最低优先级序列，重新排队

#### 2.1.6 序列状态管理

- **FR-6.1**：序列状态机：`WAITING` → `RUNNING` → `FINISHED`
- **FR-6.2**：自动递增的序列 ID
- **FR-6.3**：跟踪 prompt token 数、已缓存 token 数、已调度 token 数
- **FR-6.4**：高效的跨进程序列化（Decode 阶段仅传输最后一个 token）

---

### 2.2 优化特性

#### 2.2.1 CUDA Graph

- **FR-7.1**：Decode 阶段自动使用预捕获的 CUDA Graph
- **FR-7.2**：支持的 batch size：1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- **FR-7.3**：不同 batch size 的 Graph 共享内存池
- **FR-7.4**：可通过 `enforce_eager=True` 全局禁用

#### 2.2.2 Torch 编译

- **FR-8.1**：RMSNorm（含融合 Add 变体）使用 `@torch.compile`
- **FR-8.2**：SiluAndMul 激活函数使用 `@torch.compile`
- **FR-8.3**：RotaryEmbedding 使用 `@torch.compile`
- **FR-8.4**：Sampler 采样器使用 `@torch.compile`

#### 2.2.3 自定义 Triton 核

- **FR-9.1**：`store_kvcache_kernel`：将 K/V 张量通过 slot_mapping 写入分页 Cache
- **FR-9.2**：每个 Triton 程序实例处理一个 token 的所有 head 和维度

#### 2.2.4 张量并行（Tensor Parallelism）

- **FR-10.1**：支持 1-8 卡张量并行（NCCL 后端）
- **FR-10.2**：列并行：QKV 投影、Gate/Up 投影
- **FR-10.3**：行并行：输出投影、Down 投影（forward 中执行 all_reduce）
- **FR-10.4**：词表并行：Embedding 和 LM Head（按 vocab 维度分片）
- **FR-10.5**：进程间通信：共享内存 + pickle 序列化
- **FR-10.6**：主进程（rank 0）负责调度和输出，其余进程进入命令循环

---

### 2.3 模型支持

#### 2.3.1 Qwen3 模型

- **FR-11.1**：完整实现 Qwen3ForCausalLM 架构
- **FR-11.2**：支持 GQA（Grouped-Query Attention），num_heads 与 num_kv_heads 可不同
- **FR-11.3**：支持 QK-Norm（对 Q、K 施加 RMSNorm）
- **FR-11.4**：支持权重绑定（tie_word_embeddings）
- **FR-11.5**：HuggingFace 权重名自动映射：
  - `q_proj` / `k_proj` / `v_proj` → `qkv_proj`
  - `gate_proj` / `up_proj` → `gate_up_proj`
- **FR-11.6**：从 HuggingFace AutoConfig 自动加载模型配置

#### 2.3.2 模型扩展接口

- **FR-12.1**：新增模型只需在 `models/` 目录下创建文件
- **FR-12.2**：需实现 `packed_modules_mapping` 字典以声明权重映射关系
- **FR-12.3**：需遵循标准的 HuggingFace PreTrainedModel 接口

---

### 2.4 Tokenizer 支持

- **FR-13.1**：通过 HuggingFace AutoTokenizer 加载
- **FR-13.2**：支持 chat_template 应用
- **FR-13.3**：支持 special tokens 的正确处理

---

## 3. 非功能需求

### 3.1 性能要求

| 指标 | 要求 |
|------|------|
| 推理吞吐量 | 与 vLLM 差距 < 10%（相同硬件、相同模型、相同负载） |
| 首 Token 延迟 | Prefill 阶段无额外开销，与模型计算时间一致 |
| 显存利用率 | 可配置，默认 90% GPU 显存用于 KV Cache |
| CUDA Graph 加速 | Decode 阶段相比 eager 模式提升 20%+ |

### 3.2 可用性要求

| 指标 | 要求 |
|------|------|
| API 兼容性 | `LLM.generate()` 接口与 vLLM 基本一致 |
| 错误提示 | 权重路径无效、显存不足等场景有清晰的错误信息 |
| 进度反馈 | 长时间推理通过 tqdm 显示进度和实时吞吐量 |

### 3.3 可维护性要求

| 指标 | 要求 |
|------|------|
| 代码行数 | 核心库 <= 1500 行 |
| 单文件行数 | 单个源文件 <= 300 行 |
| 模块职责 | engine / layers / models / utils 四层分离 |
| 依赖数量 | 外部依赖 <= 5 个 |

### 3.4 兼容性要求

| 指标 | 要求 |
|------|------|
| Python 版本 | >= 3.10, < 3.13 |
| CUDA 版本 | 与 PyTorch 2.4+ 兼容 |
| 操作系统 | Linux |
| GPU | NVIDIA GPU（支持 FlashAttention） |

---

## 4. 系统架构

### 4.1 分层架构

```
┌─────────────────────────────────────────────────┐
│                   用户接口层                       │
│              LLM / SamplingParams                │
├─────────────────────────────────────────────────┤
│                   引擎层                          │
│   LLMEngine → Scheduler → ModelRunner            │
│               BlockManager   Sequence             │
├─────────────────────────────────────────────────┤
│                   模型层                          │
│            Qwen3ForCausalLM                      │
│   Attention / MLP / DecoderLayer / Embedding     │
├─────────────────────────────────────────────────┤
│                   算子层                          │
│   FlashAttention / Triton Kernel / RoPE          │
│   RMSNorm / SiluAndMul / Sampler                 │
│   Tensor-Parallel Linear / Embedding             │
├─────────────────────────────────────────────────┤
│                   基础设施                        │
│   Config / Context / WeightLoader                │
└─────────────────────────────────────────────────┘
```

### 4.2 数据流

```
用户输入 (str/list[int])
    │
    ▼
Tokenizer 编码
    │
    ▼
Sequence 创建 → Scheduler.waiting 队列
    │
    ▼
Scheduler.schedule()
    ├── Prefill: 整条 prompt 一次性处理（或分块）
    └── Decode:  每次处理一个 token/序列
    │
    ▼
ModelRunner.run()
    ├── prepare_prefill / prepare_decode: 构建输入张量
    ├── run_model: 前向推理（CUDA Graph 或 eager）
    └── Sampler: GPU 端采样 → next_token
    │
    ▼
Scheduler.postprocess()
    ├── 更新序列状态（追加 token、检测 EOS）
    ├── BlockManager.hash_blocks: 注册新块到前缀缓存
    └── 释放已完成序列的块
    │
    ▼
循环 step() 直到所有序列完成
    │
    ▼
Tokenizer 解码 → 输出文本
```

### 4.3 KV Cache 内存布局

```
KV Cache 张量: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
               │                                         
               └── 0 = Key, 1 = Value

每个注意力层获得视图:
  k_cache = kv_cache[0, layer_id]  → (num_blocks, block_size, num_kv_heads, head_dim)
  v_cache = kv_cache[1, layer_id]  → (num_blocks, block_size, num_kv_heads, head_dim)

块分配示例 (block_size=4):
  序列 token_ids = [a, b, c, d, e, f, g, h, i]
  block_table = [3, 7, 1]
  
  物理块 3: [a, b, c, d]
  物理块 7: [e, f, g, h]
  物理块 1: [i, _, _, _]  ← 当前未满
```

---

## 5. 接口定义

### 5.1 用户接口

```python
class LLM(LLMEngine):
    def __init__(
        self,
        model: str,                          # 模型路径
        max_num_batched_tokens: int = 16384, # 单批次最大 token 数
        max_num_seqs: int = 512,             # 最大并发序列数
        max_model_len: int = 4096,           # 最大序列长度
        gpu_memory_utilization: float = 0.9, # GPU 显存使用比例
        tensor_parallel_size: int = 1,       # 张量并行度
        enforce_eager: bool = False,         # 禁用 CUDA Graph
        kvcache_block_size: int = 256,       # KV Cache 块大小
    ): ...

    def generate(
        self,
        prompts: list[str] | list[list[int]],  # 文本或 token ID
        sampling_params: SamplingParams,        # 采样参数
        use_tqdm: bool = True,                  # 显示进度条
    ) -> list[dict]:  # [{"text": str, "token_ids": list[int]}]
        ...

@dataclass
class SamplingParams:
    temperature: float = 1.0     # 采样温度
    max_tokens: int = 64         # 最大生成长度
    ignore_eos: bool = False     # 是否忽略 EOS
```

### 5.2 引擎内部接口

```python
class LLMEngine:
    def add_request(prompt, sampling_params) -> None: ...
    def step() -> tuple[list[dict], int]: ...  # (outputs, num_tokens)
    def cleanup() -> None: ...

class Scheduler:
    def schedule() -> tuple[list[Sequence], bool]: ...  # (seqs, is_prefill)
    def postprocess() -> None: ...

class ModelRunner:
    def run(seqs, is_prefill) -> torch.Tensor: ...  # next_tokens
    def call(method_name, *args) -> Any: ...  # TP 跨进程调用

class BlockManager:
    def can_allocate(seq) -> int: ...      # 返回缓存命中块数，-1 表示不足
    def allocate(seq, num_cached) -> None: ...
    def can_append(seq) -> bool: ...
    def free(seq) -> None: ...
    def hash_blocks(seq) -> None: ...
```

---

## 6. 约束与限制

### 6.1 已知限制

| 编号 | 限制 | 说明 |
|------|------|------|
| L-1 | 仅支持 Qwen3 | 当前模型层仅实现 Qwen3 架构 |
| L-2 | 不支持纯贪心解码 | temperature 必须 > 1e-10 |
| L-3 | 不支持 Swap 抢占 | 抢占策略为重计算（Recompute），非换出到 CPU |
| L-4 | 仅支持 Linux | 依赖 NCCL 和 Triton |
| L-5 | 不支持流式输出 | generate() 为同步阻塞调用 |
| L-6 | 不支持动态批处理优先级 | 所有请求平等调度 |
| L-7 | 块大小必须为 256 的倍数 | Triton 核的对齐要求 |

### 6.2 技术约束

| 约束 | 说明 |
|------|------|
| FlashAttention | Prefill 使用 `flash_attn_varlen_func`，Decode 使用 `flash_attn_with_kvcache` |
| Triton | KV Cache 写入使用自定义 Triton 核，要求 triton >= 3.0.0 |
| NCCL | 张量并行依赖 NCCL 后端的 PyTorch 分布式 |
| Safetensors | 权重文件格式为 HuggingFace safetensors |

---

## 7. 验收标准

### 7.1 功能验收

- [ ] 能正确加载 Qwen3-0.6B 权重并完成推理
- [ ] 批量推理输出结果与 vLLM 语义一致（相同输入、相同随机种子）
- [ ] 前缀缓存在重复前缀场景下命中率 > 90%
- [ ] 张量并行 2/4/8 卡输出结果与单卡一致
- [ ] 显存不足时抢占机制正常工作，不 OOM

### 7.2 性能验收

- [ ] RTX 4070 + Qwen3-0.6B + 256 序列：吞吐量 >= 1300 tokens/s
- [ ] CUDA Graph 启用后 Decode 速度提升 >= 15%
- [ ] 前缀缓存命中时 Prefill 跳过已缓存块

### 7.3 代码验收

- [ ] 核心库总行数 <= 1500
- [ ] 无外部依赖超出 torch / triton / transformers / flash-attn / xxhash
- [ ] 所有公开函数有类型注解
- [ ] 无硬编码的模型路径或超参数

---

## 8. 未来演进

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 多模型支持 | Llama、DeepSeek 等架构 | 高 |
| 流式输出 | 支持 Streamer 回调 | 高 |
| CPU Offload | 支持 KV Cache 换出到 CPU 内存 | 中 |
| Speculative Decoding | 投机解码加速 | 中 |
| 量化支持 | GPTQ / AWQ / FP8 量化推理 | 中 |
| 多模态 | Vision-Language 模型支持 | 低 |
| HTTP 服务 | OpenAI 兼容 API 服务 | 低 |