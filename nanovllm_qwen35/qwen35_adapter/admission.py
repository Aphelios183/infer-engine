"""Synchronous CPU joint admission over the real nano BlockManager.

Used by the CPU HybridScheduler, not the GPU engine. No prefix reuse, chunking,
preemption, GPU execution or concurrent mutations are supported here.
"""

from nanovllm.engine.sequence import SequenceStatus


class AdmissionController:
    def __init__(self, block_manager, state_manager, *, max_prefill_tokens):
        if type(max_prefill_tokens) is not int or max_prefill_tokens <= 0:
            raise ValueError('max_prefill_tokens must be positive')
        if block_manager.block_size <= 0:
            raise ValueError('block_size must be positive')
        self.blocks = block_manager
        self.states = state_manager
        self.max_prefill_tokens = max_prefill_tokens
        self._admitted = {}

    def _snapshot(self, seq):
        # Correctness-first CPU baseline. O(total blocks) metadata copy, no KV bytes.
        bm = self.blocks
        return (list(bm.free_block_ids), set(bm.used_block_ids),
                dict(bm.hash_to_block_id),
                [(b.ref_count, b.hash, list(b.token_ids)) for b in bm.blocks],
                list(seq.block_table), seq.num_cached_tokens, seq.num_scheduled_tokens)

    def _restore(self, seq, snapshot):
        free, used, hashes, blocks, table, cached, scheduled = snapshot
        bm = self.blocks
        bm.free_block_ids.clear()
        bm.free_block_ids.extend(free)
        bm.used_block_ids.clear()
        bm.used_block_ids.update(used)
        bm.hash_to_block_id.clear()
        bm.hash_to_block_id.update(hashes)
        for block, (refs, hash_value, tokens) in zip(bm.blocks, blocks):
            block.ref_count = refs
            block.hash = hash_value
            block.token_ids = tokens
        seq.block_table[:] = table
        seq.num_cached_tokens = cached
        seq.num_scheduled_tokens = scheduled

    def try_admit(self, seq, *, token_budget=None):
        """Return state slot (0 valid), None for temporary backpressure.

        A successful admission reserves resources only: no queue transition,
        scheduled/computed progress, or forward happens in this function.
        """
        if seq.seq_id in self._admitted or seq.seq_id in self.states.owners:
            raise ValueError('duplicate request ownership')
        if (seq.status != SequenceStatus.WAITING or seq.block_table
                or seq.num_cached_tokens or seq.num_scheduled_tokens
                or seq.num_tokens != seq.num_prompt_tokens or not seq.is_prefill):
            raise ValueError('only fresh requests are supported; no resume/prefix/chunk')
        if seq.block_size != self.blocks.block_size:
            raise ValueError('Sequence and BlockManager block_size mismatch')
        if seq.num_tokens <= 0 or seq.num_tokens > self.max_prefill_tokens:
            raise ValueError('prompt cannot fit full-prefill budget; chunking disabled')
        if seq.num_blocks > len(self.blocks.blocks):
            raise ValueError('prompt exceeds total KV capacity')
        budget = self.max_prefill_tokens if token_budget is None else token_budget
        if type(budget) is not int or not 0 <= budget <= self.max_prefill_tokens:
            raise ValueError('invalid remaining token budget')
        if seq.num_tokens > budget or seq.num_blocks > len(self.blocks.free_block_ids):
            return None

        snapshot = self._snapshot(seq)
        try:
            # Never call prefix-aware can_allocate and never accept its cached count.
            self.blocks.allocate(seq, num_cached_blocks=0)
            slot = self.states.allocate(seq.seq_id)
        except BaseException:
            self._restore(seq, snapshot)
            raise
        if slot is None:
            self._restore(seq, snapshot)
            return None
        self._admitted[seq.seq_id] = seq
        return slot

    def reserve_decode(self, seq):
        """Reserve a possible next KV block, retaining the existing linear slot."""
        if self._admitted.get(seq.seq_id) is not seq:
            raise KeyError('request not owned by this coordinator')
        self.states.lookup(seq.seq_id)
        if not self.blocks.can_append(seq):
            return False
        snapshot = self._snapshot(seq)
        try:
            self.blocks.may_append(seq)
        except BaseException:
            self._restore(seq, snapshot)
            raise
        return True

    def release(self, seq):
        """Only call when request execution is quiescent; no GPU safety implied."""
        if self._admitted.get(seq.seq_id) is not seq:
            raise KeyError('request not owned by this coordinator')
        self.states.lookup(seq.seq_id)  # check before mutating either resource pool
        self.blocks.deallocate(seq)
        self.states.release(seq.seq_id)
        del self._admitted[seq.seq_id]
