"""CPU synchronous state ownership prototype; not connected to the runner.

The caller creates the pools and must not mutate other requests' slices.
CUDA is deliberately rejected until in-flight execution safety is implemented.
"""

from collections import deque

import torch


class StateManager:
    def __init__(self, conv_pool: torch.Tensor, recurrent_pool: torch.Tensor):
        # [linear_layer, request_slot, ...]; dtype is explicit in supplied pools.
        for name, pool, rank in (
            ('conv', conv_pool, 4), ('recurrent', recurrent_pool, 5)
        ):
            if not isinstance(pool, torch.Tensor):
                raise TypeError(f'{name} pool must be a tensor')
            if pool.device.type != 'cpu':
                raise ValueError('CPU-only prototype: GPU lifetime safety is not implemented')
            if pool.ndim != rank or any(d <= 0 for d in pool.shape):
                raise ValueError(f'{name} pool must have rank {rank} and positive dimensions')
            if not pool.is_floating_point() or pool.requires_grad or not pool.is_contiguous():
                raise ValueError(f'{name} pool must be contiguous floating inference storage')
        if conv_pool.shape[:2] != recurrent_pool.shape[:2]:
            raise ValueError('pools must share layer count and slot capacity')
        if conv_pool.untyped_storage().data_ptr() == recurrent_pool.untyped_storage().data_ptr():
            raise ValueError('conv and recurrent pools must not share storage')
        self.conv_pool = conv_pool
        self.recurrent_pool = recurrent_pool
        self.capacity = conv_pool.shape[1]
        self._free = deque(range(self.capacity))
        self._owners = {}

    @property
    def free_slots(self):
        return tuple(self._free)

    @property
    def owners(self):
        return dict(self._owners)

    @staticmethod
    def _validate_id(request_id):
        # nano Sequence IDs are integers; strings are useful in classroom tests.
        if type(request_id) not in (str, int) or request_id == '':
            raise ValueError('request_id must be a nonempty string or integer')

    def _initialize_slot(self, slot):
        with torch.no_grad():
            self.conv_pool[:, slot, ...].zero_()
            self.recurrent_pool[:, slot, ...].zero_()

    def allocate(self, request_id):
        """Return a slot (including 0), None if full; duplicate IDs raise.

        Initialization precedes ownership commit. On failure restore bookkeeping;
        unowned slot bytes may be partially cleared and will be reset on retry.
        """
        self._validate_id(request_id)
        if request_id in self._owners:
            raise ValueError(f'duplicate request: {request_id!r}')
        if not self._free:
            return None
        slot = self._free.popleft()
        try:
            self._initialize_slot(slot)
            self._owners[request_id] = slot
        except BaseException:
            self._free.appendleft(slot)
            raise
        return slot

    def lookup(self, request_id):
        self._validate_id(request_id)
        return self._owners[request_id]

    def release(self, request_id):
        """Caller must ensure execution is finished; this class does not cancel work.

        Keep dirty bytes until the next allocate, but remove ownership immediately.
        Request IDs must be unique for a lifecycle; stale raw slot handles are unsafe.
        """
        self._validate_id(request_id)
        slot = self._owners.pop(request_id)
        self._free.append(slot)

    def assert_consistent(self):
        all_slots = list(self._free) + list(self._owners.values())
        assert len(all_slots) == self.capacity
        assert set(all_slots) == set(range(self.capacity))
