"""Run: python -m unittest discover -s week4/test -p 'test_*.py' -v"""

import copy
import unittest
from unittest.mock import patch

from hybrid_contract import HybridResources


class HybridContractTests(unittest.TestCase):
    def test_admission_reserves_both_resources_without_processing_tokens(self):
        pool = HybridResources()
        self.assertTrue(pool.admit('A', 2))
        self.assertEqual(len(pool.free_slots), 1)
        self.assertEqual(len(pool.free_blocks), 2)
        self.assertEqual(pool.requests['A'].processed_tokens, 0)
        pool.assert_consistent()

    def test_no_state_slot_does_not_consume_available_kv(self):
        pool = HybridResources(num_slots=1)
        pool.admit('A', 1)
        before = copy.deepcopy(pool.__dict__)
        self.assertFalse(pool.admit('C', 1))
        self.assertEqual(pool.__dict__, before)

    def test_kv_shortage_rolls_back_reserved_state_slot(self):
        pool = HybridResources(num_slots=2, num_blocks=2)
        pool.admit('A', 2)
        before = copy.deepcopy(pool.__dict__)
        self.assertFalse(pool.admit('C', 1))
        self.assertEqual(pool.__dict__, before)
        self.assertNotIn('C', pool.requests)
        pool.assert_consistent()

    def test_allocation_exception_rolls_back_slot(self):
        pool = HybridResources()
        before = copy.deepcopy(pool.__dict__)
        with patch.object(pool, '_take_blocks', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                pool.admit('C', 1)
        self.assertEqual(pool.__dict__, before)

    def test_initialization_exception_returns_both_resources(self):
        pool = HybridResources()
        before = copy.deepcopy(pool.__dict__)
        with patch.object(pool, '_reset_slot', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                pool.admit('C', 1)
        self.assertEqual(pool.__dict__, before)

    def test_existing_request_updates_without_a_free_slot(self):
        pool = HybridResources(num_slots=1)
        pool.admit('A', 1)
        slot = pool.requests['A'].slot
        recurrent = pool.states[slot]['recurrent']
        blocks = list(pool.requests['A'].blocks)
        pool.fake_forward('A', 2)
        pool.fake_forward('A', 4)
        self.assertEqual(pool.free_slots, [])
        self.assertEqual(pool.requests['A'].slot, slot)
        self.assertIs(pool.states[slot]['recurrent'], recurrent)
        self.assertEqual(recurrent, [5.0, 5.0])
        self.assertEqual(pool.requests['A'].blocks, blocks)
        self.assertEqual(pool.requests['A'].processed_tokens, 2)

    def test_full_kv_only_prefix_is_rejected_before_mutation(self):
        pool = HybridResources()
        before = copy.deepcopy(pool.__dict__)
        with self.assertRaisesRegex(ValueError, 'hybrid state'):
            pool.admit('C', 1, prefix_tokens=8)
        self.assertEqual(pool.__dict__, before)

    def test_reused_slot_clears_both_states_not_only_counter(self):
        pool = HybridResources(num_slots=1)
        pool.admit('A', 1)
        pool.fake_forward('A', 10)
        slot = pool.requests['A'].slot
        pool.release('A')
        self.assertEqual(pool.states[slot]['recurrent'], [10.0, 10.0])
        pool.admit('C', 1)
        self.assertEqual(pool.requests['C'].slot, slot)
        self.assertEqual(pool.requests['C'].processed_tokens, 0)
        self.assertEqual(pool.states[slot]['conv'], [0.0, 0.0])
        self.assertEqual(pool.states[slot]['recurrent'], [0.0, 0.0])
        pool.fake_forward('C', 2)
        self.assertEqual(pool.states[slot]['recurrent'], [2.0, 2.0])

    def test_batch_reorder_preserves_request_identity(self):
        pool = HybridResources()
        pool.admit('A', 1)
        pool.admit('B', 1)
        pool.fake_forward('A', 10)
        pool.fake_forward('B', 20)
        self.assertEqual(pool.observe_batch(['B', 'A']), [(20, 20), (10, 10)])
        pool.fake_forward('B', 2)
        self.assertEqual(pool.observe_batch(['A', 'B']), [(10, 10), (12, 12)])

    def test_release_does_not_damage_neighbor_and_waiter_can_retry(self):
        pool = HybridResources(num_slots=2, num_blocks=2)
        pool.admit('A', 1)
        pool.admit('B', 1)
        pool.fake_forward('B', 20)
        b = copy.deepcopy(pool.requests['B'])
        self.assertFalse(pool.admit('C', 1))
        pool.release('A')
        self.assertTrue(pool.admit('C', 1))
        self.assertEqual(pool.requests['B'], b)
        self.assertEqual(pool.observe_batch(['B']), [(20, 20)])
        pool.assert_consistent()

    def test_duplicate_admission_and_double_release_do_not_leak(self):
        pool = HybridResources()
        pool.admit('A', 1)
        before = copy.deepcopy(pool.__dict__)
        with self.assertRaises(ValueError):
            pool.admit('A', 1)
        self.assertEqual(pool.__dict__, before)
        pool.release('A')
        before = copy.deepcopy(pool.__dict__)
        with self.assertRaises(KeyError):
            pool.release('A')
        self.assertEqual(pool.__dict__, before)
        self.assertEqual(len(pool.free_blocks), 4)
        pool.assert_consistent()

    def test_impossible_request_is_rejected_not_left_waiting_forever(self):
        pool = HybridResources(num_blocks=2)
        before = copy.deepcopy(pool.__dict__)
        with self.assertRaisesRegex(ValueError, 'total KV capacity'):
            pool.admit('C', 3)
        self.assertEqual(pool.__dict__, before)


if __name__ == '__main__':
    unittest.main()
