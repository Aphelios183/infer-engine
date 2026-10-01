"""CPU-only teaching model. NOT nano-vLLM integration or Qwen3.5 math."""

from dataclasses import dataclass


@dataclass
class RequestState:
    slot: int
    blocks: list[int]
    processed_tokens: int = 0


class HybridResources:
    """A synchronous resource transaction, with one illustrative state per request.

    Real Qwen3.5 needs per-layer tensors; lists here only expose ownership bugs.
    State slots and KV blocks are deliberately separate resource pools.
    """

    def __init__(self, num_slots=2, num_blocks=4):
        self.num_blocks = num_blocks
        self.free_slots = list(range(num_slots))
        self.free_blocks = list(range(num_blocks))
        self.states = [dict(conv=[0.0, 0.0], recurrent=[0.0, 0.0])
                       for _ in range(num_slots)]
        self.requests: dict[str, RequestState] = {}

    def _reset_slot(self, slot):
        for values in self.states[slot].values():
            values[:] = [0.0] * len(values)

    def _take_blocks(self, count):
        if count > len(self.free_blocks):
            return None
        blocks = self.free_blocks[:count]
        del self.free_blocks[:count]
        return blocks

    def admit(self, request_id, block_count, *, prefix_tokens=0):
        """True=admitted; False=temporarily unavailable; invalid input raises.

        Simplification: no prefix reuse is supported, even if Full KV is present.
        The caller already checked compute budget. No real forward is performed.
        """
        if request_id in self.requests:
            raise ValueError('duplicate request')
        if block_count <= 0 or prefix_tokens < 0:
            raise ValueError('invalid resource request')
        if prefix_tokens:
            raise ValueError('prefix reuse requires matching hybrid state; disabled')
        if block_count > len(self.free_blocks) + sum(
            len(r.blocks) for r in self.requests.values()
        ):
            raise ValueError('request exceeds total KV capacity')
        if not self.free_slots:
            return False

        # Reserve a slot first, then KV. Commit ownership only when both succeed.
        slot = self.free_slots.pop(0)
        blocks = None
        try:
            blocks = self._take_blocks(block_count)
            if blocks is None:
                return False
            self._reset_slot(slot)  # reused memory must not become reused history
            self.requests[request_id] = RequestState(slot, blocks)
            return True
        finally:
            if request_id not in self.requests:
                if blocks is not None:
                    self.free_blocks.extend(blocks)
                    self.free_blocks.sort()
                self.free_slots.append(slot)
                self.free_slots.sort()

    def observe_batch(self, request_ids):
        # Read-only copies: batch order must not become state ownership.
        return [tuple(self.states[self.requests[r].slot]['recurrent'])
                for r in request_ids]

    def fake_forward(self, request_id, token):
        """Illustrative recurrence only, NOT Gated DeltaNet or a GPU forward.

        No new KV block is needed in this toy step: caller assumes capacity fits.
        """
        req = self.requests[request_id]
        state = self.states[req.slot]
        state['conv'][:] = [state['conv'][1], float(token)]
        state['recurrent'][:] = [x * 0.5 + token for x in state['recurrent']]
        req.processed_tokens += 1

    def release(self, request_id):
        req = self.requests.pop(request_id)  # unknown/double release must fail
        self.free_blocks.extend(req.blocks)
        self.free_blocks.sort()
        self.free_slots.append(req.slot)
        self.free_slots.sort()
        # Deliberately retain physical data. admit() must reset before reuse.

    def assert_consistent(self):
        owned_slots = [r.slot for r in self.requests.values()]
        owned_blocks = [b for r in self.requests.values() for b in r.blocks]
        all_slots = self.free_slots + owned_slots
        all_blocks = self.free_blocks + owned_blocks
        assert len(all_slots) == len(set(all_slots)) == len(self.states)
        assert len(all_blocks) == len(set(all_blocks)) == self.num_blocks
        assert not set(self.free_slots) & set(owned_slots)
        assert not set(self.free_blocks) & set(owned_blocks)
