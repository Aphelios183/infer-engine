"""Explicit single-GPU eager Qwen3.5 entry; upstream Qwen3 LLM is unchanged."""
from qwen35_adapter.gpu_engine import Qwen35Engine as Qwen35LLM

__all__ = ['Qwen35LLM']
