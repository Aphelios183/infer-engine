"""Load the GPU engine only when LLM is requested, not for CPU resource tests."""

from nanovllm.sampling_params import SamplingParams

__all__ = ['LLM', 'SamplingParams']


def __getattr__(name):
    if name == 'LLM':
        from nanovllm.llm import LLM
        globals()['LLM'] = LLM
        return LLM
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
