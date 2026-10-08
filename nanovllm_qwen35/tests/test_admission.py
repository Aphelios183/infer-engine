import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.state_manager import StateManager


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.block_size_patch = patch.object(Sequence, 'block_size', 4)
        self.block_size_patch.start()
        self.addCleanup(self.block_size_patch.stop)
        self.bm = BlockManager(4, 4)
        self.sm = StateManager(torch.zeros(2, 2, 3, 2), torch.zeros(2, 2, 1, 3, 2))
        self.admission = AdmissionController(self.bm, self.sm, max_prefill_tokens=16)

    def seq(self, n=5):
        return Sequence(list(range(n)))

    def snapshot(self, seq):
        return (self.admission._snapshot(seq), self.sm.owners, self.sm.free_slots,
                seq.status, list(seq.token_ids))

    def assert_conserved(self):
        owned = [b for seq in self.admission._admitted.values() for b in seq.block_table]
        self.assertEqual(set(owned), self.bm.used_block_ids)
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned) | set(self.bm.free_block_ids), set(range(4)))
        self.assertEqual(set(self.sm.owners), set(self.admission._admitted))
        self.sm.assert_consistent()

    def test_real_classes_and_zero_slot_success_without_progress(self):
        seq = self.seq()
        self.assertEqual(self.admission.try_admit(seq), 0)
        self.assertEqual(seq.block_table, [0, 1])
        self.assertEqual((seq.num_cached_tokens, seq.num_scheduled_tokens), (0, 0))
        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assert_conserved()

    def test_state_full_after_kv_allocation_restores_snapshot(self):
        self.admission.try_admit(self.seq(1))
        self.admission.try_admit(self.seq(1))
        seq = self.seq(1)
        before = self.snapshot(seq)
        with patch.object(self.bm, 'allocate', wraps=self.bm.allocate) as allocate:
            self.assertIsNone(self.admission.try_admit(seq))
            allocate.assert_called_once()
        self.assertEqual(self.snapshot(seq), before)
        self.assert_conserved()

    def test_kv_shortage_does_not_reserve_state(self):
        self.admission.try_admit(self.seq(16))
        seq = self.seq(1)
        before = self.snapshot(seq)
        self.assertIsNone(self.admission.try_admit(seq))
        self.assertEqual(self.snapshot(seq), before)

    def test_partial_kv_allocation_exception_rolls_back(self):
        seq = self.seq()
        before = self.snapshot(seq)
        original = self.bm._allocate_block
        count = 0
        def fail_second():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('injected KV failure')
            return original()
        with patch.object(self.bm, '_allocate_block', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'injected KV'):
                self.admission.try_admit(seq)
        self.assertEqual(self.snapshot(seq), before)
        self.assert_conserved()

    def test_state_initialization_exception_returns_kv_and_slot(self):
        seq = self.seq()
        before = self.snapshot(seq)
        with patch.object(self.sm, '_initialize_slot', side_effect=RuntimeError('init')):
            with self.assertRaises(RuntimeError):
                self.admission.try_admit(seq)
        self.assertEqual(self.snapshot(seq), before)
        self.assert_conserved()

    def test_no_full_attention_only_prefix_reuse(self):
        first, second = self.seq(), self.seq()
        self.admission.try_admit(first)
        first.num_scheduled_tokens = 5
        self.bm.hash_blocks(first)
        first.num_scheduled_tokens = 0
        self.assertEqual(self.bm.can_allocate(second), 1)  # upstream would hit a prefix
        self.admission.try_admit(second)
        self.assertTrue(set(first.block_table).isdisjoint(second.block_table))
        self.assertEqual(second.num_cached_tokens, 0)
        self.assert_conserved()

    def test_evicted_free_prefix_metadata_restored_on_failure(self):
        old = self.seq(16)
        self.admission.try_admit(old)
        old.num_scheduled_tokens = 16
        self.bm.hash_blocks(old)
        old.num_scheduled_tokens = 0
        self.admission.release(old)
        seq = self.seq(16)
        before = self.snapshot(seq)
        with patch.object(self.sm, '_initialize_slot', side_effect=RuntimeError('init')):
            with self.assertRaises(RuntimeError):
                self.admission.try_admit(seq)
        self.assertEqual(self.snapshot(seq), before)

    def test_remaining_budget_waits_but_impossible_prompt_raises(self):
        seq = self.seq()
        before = self.snapshot(seq)
        self.assertIsNone(self.admission.try_admit(seq, token_budget=4))
        self.assertEqual(self.snapshot(seq), before)
        with self.assertRaisesRegex(ValueError, 'chunking disabled'):
            self.admission.try_admit(self.seq(17))
        small = AdmissionController(BlockManager(1, 4), self.sm, max_prefill_tokens=16)
        with self.assertRaisesRegex(ValueError, 'total KV capacity'):
            small.try_admit(seq)

    def test_rejects_mismatched_blocks_and_nonfresh_request(self):
        seq = self.seq()
        seq.block_size = 8
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.admission.try_admit(seq)
        for field, value in [('num_cached_tokens', 1), ('num_scheduled_tokens', 1),
                             ('is_prefill', False), ('status', SequenceStatus.RUNNING)]:
            seq = self.seq()
            setattr(seq, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admission.try_admit(seq)

    def test_release_retries_neighbor_isolation_duplicate_and_forgery(self):
        a, b, c = self.seq(), self.seq(), self.seq()
        self.admission.try_admit(a)
        b_slot = self.admission.try_admit(b)
        self.sm.recurrent_pool[:, b_slot].fill_(9)
        b_blocks = list(b.block_table)
        self.assertIsNone(self.admission.try_admit(c))
        with self.assertRaises(ValueError):
            self.admission.try_admit(a)
        forged = self.seq()
        forged.seq_id = a.seq_id
        with self.assertRaises(KeyError):
            self.admission.release(forged)
        self.admission.release(a)
        with self.assertRaises(KeyError):
            self.admission.release(a)
        self.assertIsNotNone(self.admission.try_admit(c))
        self.assertEqual(b.block_table, b_blocks)
        self.assertTrue(torch.all(self.sm.recurrent_pool[:, b_slot] == 9))
        self.assert_conserved()


if __name__ == '__main__':
    unittest.main()
