"""Registered Qwen3.5 text parameter skeleton, deliberately WITHOUT forward."""
from copy import deepcopy

import torch
from torch import nn

from .weight_plan import expected_text


class ParameterNode(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError('parameter storage only; Qwen3.5 execution is not implemented')


class Qwen35TextParameters(ParameterNode):
    """Meta-first storage tree; names follow the separate-projection text plan.

    This is not a functional Transformer. No random initialization, cache, CUDA,
    packed projection or dtype conversion. Loader replaces every meta parameter.
    """
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        self.loaded = False
        self.model = ParameterNode()
        self.model.layers = nn.ModuleList(
            [ParameterNode() for _ in config['text_config']['layer_types']])
        self.lm_head = ParameterNode()
        self.parameter_semantics = {}
        for spec in expected_text(config).values():
            path = spec['target']
            parts = path.split('.')
            node = self
            for part in parts[:-1]:
                if part not in node._modules:
                    node.add_module(part, ParameterNode())
                node = node._modules[part]
            node.register_parameter(parts[-1], nn.Parameter(
                torch.empty(spec['shape'], device='meta'), requires_grad=False))
            self.parameter_semantics[path] = spec['semantics']
        self.tie_weights()

    def tie_weights(self):
        # The SAME Parameter object, not merely equal-valued tensors.
        self.lm_head.weight = self.model.embed_tokens.weight

    def replace_parameter(self, name, tensor):
        old = self.get_parameter(name)
        if tuple(tensor.shape) != tuple(old.shape):
            raise ValueError(f'parameter shape mismatch: {name}')
        parent, leaf = name.rsplit('.', 1)
        self.get_submodule(parent).register_parameter(
            leaf, nn.Parameter(tensor, requires_grad=False))
