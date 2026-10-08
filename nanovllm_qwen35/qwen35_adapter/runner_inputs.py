"""Prepare explicit CPU inputs from a live HybridScheduler batch.

No model, CUDA transfer, context mutation, sampling, or resource allocation.
Do not pass this generic metadata object directly to an upstream attention kernel.
"""

from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import SequenceStatus


@dataclass(frozen=True)
class RunnerInputs:
    request_ids: tuple
    is_prefill: bool
    input_ids: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    state_slots: torch.Tensor
    last_indices: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class RunnerInputBuilder:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def prepare(self, batch):
        """Read a still-pending batch; return fresh CPU tensors without advancing it.

        Preparation errors leave the pending batch and resources intact. Since no
        forward ran here, caller may explicitly fail_batch after confirming idle.
        Output tensors are snapshots, not aliases into state pools. They are mutable
        torch tensors despite the frozen dataclass, and must be treated read-only.
        """
        scheduler = self.scheduler
        scheduler._check_batch(batch)  # shared pending-batch identity guard
        admission = scheduler.admission
        bm, sm = admission.blocks, admission.states
        n = len(batch.sequences)
        if n == 0 or len(batch.input_counts) != n or len(batch.state_slots) != n:
            raise ValueError('inconsistent batch metadata lengths')
        ids, positions, slots, tables, lengths = [], [], [], [], []
        q_edges, k_edges, last = [0], [0], []
        request_ids, state_slots = [], []
        seen_blocks, seen_requests = set(), set()
        for seq, count, slot in zip(batch.sequences, batch.input_counts, batch.state_slots):
            if seq.seq_id in seen_requests or admission._admitted.get(seq.seq_id) is not seq:
                raise ValueError('duplicate or unowned request')
            seen_requests.add(seq.seq_id)
            if seq.status != SequenceStatus.RUNNING or seq not in scheduler.running:
                raise ValueError('request is not running')
            if sm.lookup(seq.seq_id) != slot:
                raise ValueError('state slot ownership mismatch')
            if seq.block_size != bm.block_size:
                raise ValueError('block size mismatch')
            if type(count) is not int or count <= 0 or seq.num_scheduled_tokens != count:
                raise ValueError('invalid scheduled count')
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + count
            if seq.num_tokens != len(seq.token_ids) or end != seq.num_tokens:
                raise ValueError('input range does not match sequence length')
            if batch.is_prefill:
                if start != 0 or seq.num_tokens != seq.num_prompt_tokens or not seq.is_prefill:
                    raise ValueError('only fresh full prefill is supported')
            elif count != 1 or start != seq.num_tokens - 1 or seq.is_prefill:
                raise ValueError('decode must process exactly one new token')
            if seq.last_token != seq.token_ids[-1]:
                raise ValueError('last_token disagrees with token_ids')
            required = (end + bm.block_size - 1) // bm.block_size
            if len(seq.block_table) != required:
                raise ValueError('missing or unexpected KV block capacity')
            for block_id in seq.block_table:
                if type(block_id) is not int or not 0 <= block_id < len(bm.blocks):
                    raise ValueError('invalid physical block ID')
                if (block_id in seen_blocks or block_id not in bm.used_block_ids
                        or bm.blocks[block_id].ref_count != 1):
                    raise ValueError('shared, duplicate or unallocated KV block')
                seen_blocks.add(block_id)
            tokens = seq.token_ids[start:end]
            if any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError('token IDs must be nonnegative integers')
            ids.extend(tokens)
            positions.extend(range(start, end))
            slots.extend(seq.block_table[p // bm.block_size] * bm.block_size
                         + p % bm.block_size for p in range(start, end))
            tables.append(list(seq.block_table))
            lengths.append(end)
            q_edges.append(q_edges[-1] + count)
            k_edges.append(k_edges[-1] + end)
            last.append(q_edges[-1] - 1)
            request_ids.append(seq.seq_id)
            state_slots.append(slot)
        if len(set(state_slots)) != n:
            raise ValueError('state slots must be unique for distinct requests')
        width = max(map(len, tables))
        padded_tables = [row + [-1] * (width - len(row)) for row in tables]
        def tensor(data, dtype=torch.int32):
            return torch.tensor(data, dtype=dtype, device='cpu')
        return RunnerInputs(
            tuple(request_ids), batch.is_prefill,
            tensor(ids, torch.int64), tensor(positions, torch.int64),
            tensor(q_edges), tensor(k_edges), tensor(lengths), tensor(slots),
            tensor(padded_tables), tensor(state_slots), tensor(last, torch.int64),
            max(batch.input_counts), max(lengths))
