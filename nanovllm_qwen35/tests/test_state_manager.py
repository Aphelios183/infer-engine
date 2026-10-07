import unittest
from unittest.mock import patch

import torch

from qwen35_adapter.state_manager import StateManager


def make_manager(capacity=2):
    return StateManager(torch.full((2, capacity, 3, 2), 9.0),
                        torch.full((2, capacity, 2, 3, 4), 7.0))


class StateManagerTests(unittest.TestCase):
    def test_zero_is_success_and_both_slices_are_initialized(self):
        m = make_manager()
        slot = m.allocate('A')
        self.assertIsNotNone(slot)
        self.assertEqual(slot, 0)
        self.assertEqual(m.lookup('A'), 0)
        self.assertEqual(torch.count_nonzero(m.conv_pool[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(m.recurrent_pool[:, 0]).item(), 0)
        self.assertTrue(torch.all(m.conv_pool[:, 1] == 9))
        self.assertTrue(torch.all(m.recurrent_pool[:, 1] == 7))
        m.assert_consistent()

    def test_full_pool_changes_nothing(self):
        m = make_manager(1)
        m.allocate('A')
        owners, free = m.owners, m.free_slots
        conv, recurrent = m.conv_pool.clone(), m.recurrent_pool.clone()
        self.assertIsNone(m.allocate('C'))
        self.assertEqual((m.owners, m.free_slots), (owners, free))
        self.assertTrue(torch.equal(conv, m.conv_pool))
        self.assertTrue(torch.equal(recurrent, m.recurrent_pool))

    def test_duplicate_even_when_full(self):
        m = make_manager(1)
        m.allocate('A')
        with self.assertRaises(ValueError):
            m.allocate('A')
        self.assertEqual(m.owners, {'A': 0})
        m.assert_consistent()

    def test_unknown_lookup_and_double_release(self):
        m = make_manager()
        with self.assertRaises(KeyError):
            m.lookup('missing')
        with self.assertRaises(KeyError):
            m.release('missing')
        m.allocate('A')
        m.release('A')
        before = m.free_slots
        with self.assertRaises(KeyError):
            m.release('A')
        self.assertEqual(m.free_slots, before)
        m.assert_consistent()

    def test_reuse_clears_all_layers_without_touching_neighbor(self):
        m = make_manager()
        a, b = m.allocate('A'), m.allocate('B')
        m.conv_pool[:, a].fill_(11)
        m.recurrent_pool[:, a].fill_(12)
        m.conv_pool[:, b].fill_(21)
        m.recurrent_pool[:, b].fill_(22)
        pointers = (m.conv_pool.data_ptr(), m.recurrent_pool.data_ptr())
        m.release('A')
        self.assertTrue(torch.all(m.recurrent_pool[:, a] == 12))
        self.assertEqual(m.allocate('C'), a)
        self.assertTrue(torch.all(m.conv_pool[:, a] == 0))
        self.assertTrue(torch.all(m.recurrent_pool[:, a] == 0))
        self.assertTrue(torch.all(m.conv_pool[:, b] == 21))
        self.assertTrue(torch.all(m.recurrent_pool[:, b] == 22))
        self.assertEqual(m.lookup('B'), b)
        self.assertEqual(pointers, (m.conv_pool.data_ptr(), m.recurrent_pool.data_ptr()))
        m.assert_consistent()

    def test_lookup_preserves_history_and_batch_order(self):
        m = make_manager()
        m.allocate('A')
        m.allocate('B')
        m.conv_pool[:, 0].fill_(11)
        self.assertEqual([m.lookup(r) for r in ['B', 'A']], [1, 0])
        self.assertTrue(torch.all(m.conv_pool[:, 0] == 11))

    def test_partial_initialization_failure_rolls_back_ownership(self):
        m = make_manager()
        m.allocate('B')
        m.conv_pool[:, 0].fill_(21)
        before = (m.owners, m.free_slots)
        def fail(slot):
            m.conv_pool[:, slot].zero_()
            raise RuntimeError('injected initialization failure')
        with patch.object(m, '_initialize_slot', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                m.allocate('C')
        self.assertEqual((m.owners, m.free_slots), before)
        self.assertTrue(torch.all(m.conv_pool[:, 0] == 21))
        self.assertEqual(m.allocate('C'), 1)
        self.assertTrue(torch.all(m.recurrent_pool[:, 1] == 0))
        m.assert_consistent()

    def test_snapshot_accessors_do_not_expose_internal_ownership(self):
        m = make_manager()
        m.allocate(0)
        snapshot = m.owners
        snapshot.clear()
        self.assertEqual(m.lookup(0), 0)

    def test_invalid_request_ids(self):
        m = make_manager()
        for value in ('', None, True, [], 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.allocate(value)
        m.assert_consistent()

    def test_invalid_shapes_and_storage(self):
        cases = [
            (torch.zeros(2, 2, 3), torch.zeros(2, 2, 2, 3, 4)),
            (torch.zeros(2, 0, 3, 2), torch.zeros(2, 0, 2, 3, 4)),
            (torch.zeros(2, 2, 3, 2), torch.zeros(2, 3, 2, 3, 4)),
            (torch.zeros(2, 2, 3, 2, dtype=torch.int64), torch.zeros(2, 2, 2, 3, 4)),
            (torch.zeros(2, 2, 3, 2, requires_grad=True), torch.zeros(2, 2, 2, 3, 4)),
            (torch.zeros(2, 2, 3, 2).transpose(2, 3), torch.zeros(2, 2, 2, 3, 4)),
        ]
        for conv, recurrent in cases:
            with self.subTest(shape=conv.shape), self.assertRaises(ValueError):
                StateManager(conv, recurrent)

    def test_aliasing_and_non_cpu_rejected(self):
        base = torch.zeros(24)
        with self.assertRaisesRegex(ValueError, 'share storage'):
            StateManager(base.view(2, 2, 3, 2), base.view(2, 2, 1, 3, 2))
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            StateManager(torch.empty(2, 2, 3, 2, device='meta'),
                         torch.empty(2, 2, 1, 3, 2, device='meta'))

    def test_explicit_dtypes_are_preserved(self):
        m = StateManager(torch.ones(2, 2, 3, 2, dtype=torch.bfloat16),
                         torch.ones(2, 2, 1, 3, 2, dtype=torch.float32))
        m.allocate('A')
        self.assertEqual(m.conv_pool.dtype, torch.bfloat16)
        self.assertEqual(m.recurrent_pool.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
