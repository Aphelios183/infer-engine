import os
import json
from dataclasses import dataclass
from transformers import AutoConfig
from qwen35_adapter.contracts import guard_unimplemented_model


@dataclass(slots=True)
class Config:
    model: str#模型路径
    max_num_batched_tokens: int = 16384#一次推理最多的总tokens
    max_num_seqs: int = 512#同时允许存在的最大请求序列数量
    max_model_len: int = 4096#单个请求最多能生成的输入+输出总长度
    gpu_memory_utilization: float = 0.9#显存使用率上限
    tensor_parallel_size: int = 1#模型可切多少张GPU
    enforce_eager: bool = False#是否强制使用Eager模式（关闭cuda graph加速）
    device: str = "cuda"
    hf_config: AutoConfig | None = None #hf格式的模型配置对象
    eos: int = -1#eos是句子结束符的token id -1为尚未设置，后面会更新正确
    kvcache_block_size: int = 256#kvcache的大小（每个块存的token键值对）
    num_kvcache_blocks: int = -1#kvcache的总块数-1为尚未设置，后面会更新正确

    def __post_init__(self):
        assert os.path.isdir(self.model) #强制传入路径为本地模型
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        with open(os.path.join(self.model, "config.json"), encoding="utf-8") as handle:
            guard_unimplemented_model(json.load(handle))
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)#引擎内部只会处理不超过这个最大长度的请求。


