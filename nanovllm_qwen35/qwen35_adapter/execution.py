"""CPU synchronous execution seam. No Qwen3.5 backend or GPU support yet."""

from dataclasses import dataclass
from typing import Protocol

import torch

from .runner_inputs import RunnerInputBuilder, RunnerInputs


class CPUBackend(Protocol):
    """Trusted synchronous backend: no background work, even on exceptions.

    Own/bind storage at construction. forward consumes KV addresses and state_slots
    from inputs; it must not allocate/release request resources or alter counters.
    Both methods return CPU floating tensors. No CUDA work is permitted.
    """
    execution_mode: str  # must be 'cpu_sync'

    def forward(self, inputs: RunnerInputs) -> torch.Tensor:
        """Return packed hidden states [total_tokens, hidden_size]."""
        ...

    def compute_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Return raw logits [requests, vocab_size], NOT probabilities or IDs."""
        ...


class CPUSampler(Protocol):
    execution_mode: str  # same no-background-work contract as CPUBackend

    def sample(self, logits: torch.Tensor, temperatures: tuple) -> torch.Tensor:
        """Return CPU int64 token IDs [requests], in unchanged request order."""
        ...


@dataclass(frozen=True)
class StepResult:
    request_ids: tuple
    token_ids: tuple
    input_counts: tuple
    is_prefill: bool


class CPUExecutionRunner:
    """Execute AND settle one pending batch; caller must not settle it twice.

    Serialized owner-thread use only. Preparation errors retain the batch (metadata
    may need repair). Once execution starts, an error fails all touched requests;
    this is disposal, not rollback. CPU sync is a trusted implementation contract,
    not a runtime detector for hidden asynchronous GPU work.
    """
    def __init__(self, scheduler, backend: CPUBackend, sampler: CPUSampler):
        for component in (backend, sampler):
            if getattr(component, 'execution_mode', None) != 'cpu_sync':
                raise ValueError('only explicit cpu_sync implementations are supported')
        self.scheduler = scheduler
        self.builder = RunnerInputBuilder(scheduler)
        self.backend = backend
        self.sampler = sampler

    @staticmethod
    def _matrix(value, rows, name):
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.ndim != 2 or value.shape[0] != rows or value.shape[1] <= 0
                or not value.is_floating_point()):
            raise ValueError(f'{name} must be a CPU floating matrix with {rows} rows')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'{name} contains NaN/Inf')

    @torch.inference_mode()
    def run(self, batch):
        inputs = self.builder.prepare(batch)  # outside cleanup: no execution yet
        temperatures = tuple(seq.temperature for seq in batch.sequences)
        try:
            hidden = self.backend.forward(inputs)
            self._matrix(hidden, inputs.input_ids.numel(), 'hidden')
            last_hidden = hidden.index_select(0, inputs.last_indices)
            logits = self.backend.compute_logits(last_hidden)
            n = len(inputs.request_ids)
            self._matrix(logits, n, 'logits')
            ids = self.sampler.sample(logits, temperatures)
            if (not isinstance(ids, torch.Tensor) or ids.device.type != 'cpu'
                    or ids.dtype != torch.int64 or ids.shape != (n,)):
                raise ValueError('sampler must return CPU int64 [requests]')
            if ((ids < 0) | (ids >= logits.shape[1])).any().item():
                raise ValueError('sampled token outside vocabulary')
            token_ids = tuple(ids.tolist())
        except BaseException as exc:
            # Safe ONLY under the documented synchronous CPU contract.
            self.scheduler.fail_batch(batch, f'execution: {exc}', execution_complete=True)
            raise
        # Settlement errors are not execution errors: never attempt a second release.
        self.scheduler.postprocess(batch, token_ids)
        return StepResult(inputs.request_ids, token_ids, batch.input_counts, batch.is_prefill)
