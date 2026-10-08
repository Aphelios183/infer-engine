"""CPU synchronous hybrid scheduling over real nano Sequence/resource managers.

No model execution, GPU safety, preemption, prefix reuse or chunked prefill.
One outstanding batch; callers must settle or explicitly fail it before scheduling.
"""

from collections import deque
from dataclasses import dataclass

from nanovllm.engine.sequence import SequenceStatus


class NoProgressError(RuntimeError):
    """Resources are retained; caller must explicitly fail/cancel to recover."""


@dataclass(frozen=True)
class ScheduledBatch:#已调度batch
    sequences: tuple
    is_prefill: bool
    state_slots: tuple
    input_counts: tuple

    @property
    def total_tokens(self):
        return sum(self.input_counts)


@dataclass(frozen=True)
class Completion:#请求完成时的结果
    request_id: int
    reason: str
    token_ids: tuple
    processed_tokens: int
    error: str | None = None


class HybridScheduler:
    def __init__(self, admission, *, max_num_seqs, token_budget, eos_ids):
        if type(max_num_seqs) is not int or max_num_seqs <= 0:
            raise ValueError('max_num_seqs must be positive')
        if type(token_budget) is not int or not 0 < token_budget <= admission.max_prefill_tokens:
            raise ValueError('token_budget must fit admission full-prefill limit')
        self.admission = admission
        self.max_num_seqs = max_num_seqs  # cap both active requests and each batch
        self.token_budget = token_budget
        self.eos_ids = frozenset(eos_ids)
        self.waiting = deque()
        self.running = deque()
        self.completed = {}
        self._known_ids = set()
        self._pending = None

    def add(self, seq):
        if seq.seq_id in self._known_ids or seq.seq_id in self.admission.states.owners:
            raise ValueError('duplicate request ID')
        if (seq.status != SequenceStatus.WAITING or seq.block_table
                or seq.num_cached_tokens or seq.num_scheduled_tokens
                or not seq.is_prefill or seq.num_tokens != seq.num_prompt_tokens):
            raise ValueError('only fresh requests can be submitted')
        if type(seq.max_tokens) is not int or seq.max_tokens < 0:
            raise ValueError('max_tokens must be a nonnegative integer')
        if seq.block_size != self.admission.blocks.block_size:
            raise ValueError('block size mismatch')
        if seq.max_tokens and (seq.num_tokens > self.token_budget
                              or seq.num_blocks > len(self.admission.blocks.blocks)):
            raise ValueError('prompt cannot fit: chunking/preemption disabled')
        self._known_ids.add(seq.seq_id)
        self.waiting.append(seq)
        if seq.max_tokens == 0:
            self._finish(seq, 'length')  # no forward, no resource allocation

    def _finish(self, seq, reason, error=None):
        processed = seq.num_cached_tokens
        if seq in self.running:
            self.admission.release(seq)
            self.running.remove(seq)
        else:
            self.waiting.remove(seq)
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.FINISHED
        self.completed[seq.seq_id] = Completion(
            seq.seq_id, reason, tuple(seq.token_ids), processed, error)

    def _make_batch(self, selected, is_prefill):
        counts = tuple(s.num_tokens if is_prefill else 1 for s in selected)
        slots = tuple(self.admission.states.lookup(s.seq_id) for s in selected)
        for seq, count in zip(selected, counts):
            seq.num_scheduled_tokens = count
            seq.is_prefill = is_prefill
        self._pending = ScheduledBatch(tuple(selected), is_prefill, slots, counts)
        return self._pending

    def schedule(self):
        if self._pending is not None:
            raise RuntimeError('previous batch is not settled')
        # Homogeneous batches: prioritize existing decode; rotate selected requests.
        selected = []
        for seq in tuple(self.running):
            if len(selected) >= min(self.max_num_seqs, self.token_budget):
                break
            try:
                if seq.num_cached_tokens != seq.num_tokens - 1 or seq.num_scheduled_tokens:
                    raise ValueError('invalid decode counters')
                available = self.admission.reserve_decode(seq)
            except Exception as exc:
                self._finish(seq, 'failed', f'decode reservation: {exc}')
                continue
            if available:
                selected.append(seq)
                self.running.remove(seq)
                self.running.append(seq)
        if selected:
            return self._make_batch(selected, False)

        budget = self.token_budget
        for seq in tuple(self.waiting):
            if len(self.running) >= self.max_num_seqs:
                break
            try:
                slot = self.admission.try_admit(seq, token_budget=budget)
            except Exception as exc:
                self._finish(seq, 'failed', f'admission: {exc}')
                continue
            if slot is None:
                continue  # scan later requests; do not mutate this waiting request
            self.waiting.remove(seq)
            self.running.append(seq)
            seq.status = SequenceStatus.RUNNING  # scheduled, not executed yet
            selected.append(seq)
            budget -= seq.num_tokens
            if budget == 0:
                break
        if selected:
            return self._make_batch(selected, True)
        if self.waiting or self.running:
            raise NoProgressError(
                'no runnable request; no preemption implemented; resources retained, '
                'explicitly cancel/fail a request or revise capacity policy')
        return None

    def _check_batch(self, batch):
        if batch is not self._pending or batch is None:
            raise ValueError('stale or foreign batch')

    def postprocess(self, batch, token_ids):
        """Call only after synchronous forward AND sampling completed successfully."""
        self._check_batch(batch)
        try:
            tokens = tuple(token_ids)
            if len(tokens) != len(batch.sequences) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError('exactly one nonnegative integer token per request required')
        except Exception:
            # Model states may already have changed; never retry this batch on dirty state.
            self.fail_batch(batch, 'invalid model output', execution_complete=True)
            raise
        for seq, token, count in zip(batch.sequences, tokens, batch.input_counts):
            seq.num_cached_tokens += count
            seq.num_scheduled_tokens = 0
            seq.append_token(token)
            if not seq.ignore_eos and token in self.eos_ids:
                self._finish(seq, 'eos')
            elif seq.num_completion_tokens >= seq.max_tokens:
                self._finish(seq, 'length')
        self._pending = None

    def fail_batch(self, batch, error, *, execution_complete=False):
        """No tensor rollback: fail all possibly touched requests after quiescence."""
        self._check_batch(batch)
        if not execution_complete:
            raise RuntimeError('cannot recycle resources until execution has stopped')
        for seq in batch.sequences:
            self._finish(seq, 'failed', str(error))
        self._pending = None

    def fail(self, request_id, error='request failed'):
        self._terminate_idle(request_id, 'failed', str(error))

    def cancel(self, request_id):
        self._terminate_idle(request_id, 'cancelled', None)

    def _terminate_idle(self, request_id, reason, error):
        if self._pending is not None and any(s.seq_id == request_id for s in self._pending.sequences):
            raise RuntimeError('request belongs to pending batch; settle or fail_batch after quiescence')
        for seq in (*self.waiting, *self.running):
            if seq.seq_id == request_id:
                self._finish(seq, reason, error)
                return
        raise KeyError(request_id)

    def is_finished(self):
        return not self.waiting and not self.running and self._pending is None
