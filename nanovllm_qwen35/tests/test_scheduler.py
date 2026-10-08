import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.state_manager import StateManager
from qwen35_adapter.scheduler import HybridScheduler, NoProgressError


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(Sequence, 'block_size', 4)
        p.start()
        self.addCleanup(p.stop)

    def make(self, blocks=8, slots=3, budget=16, max_seqs=3):
        bm = BlockManager(blocks, 4)
        sm = StateManager(torch.zeros(2, slots, 3, 2), torch.zeros(2, slots, 1, 3, 2))
        admission = AdmissionController(bm, sm, max_prefill_tokens=budget)
        return HybridScheduler(admission, max_num_seqs=max_seqs, token_budget=budget, eos_ids={99})

    def seq(self, n=3, max_tokens=3, ignore_eos=False):
        return Sequence(list(range(n)), SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))

    def assert_resources(self, s):
        a = s.admission
        slots = a.states.owners
        self.assertEqual(set(slots), {r.seq_id for r in s.running})
        self.assertEqual(set(a._admitted), set(slots))
        used = [b for r in s.running for b in r.block_table]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(set(used), a.blocks.used_block_ids)
        self.assertEqual(set(used) | set(a.blocks.free_block_ids), set(range(len(a.blocks.blocks))))
        self.assertTrue(set(used).isdisjoint(a.blocks.free_block_ids))
        for block in a.blocks.blocks:
            self.assertEqual(block.ref_count, int(block.block_id in used))
        for r in s.waiting:
            self.assertEqual((r.block_table, r.num_cached_tokens, r.num_scheduled_tokens), ([], 0, 0))
        a.states.assert_consistent()

    def test_two_round_lifecycle_preserves_output_and_recycles(self):
        s = self.make()
        a = self.seq(max_tokens=2)
        s.add(a)
        batch = s.schedule()
        self.assertTrue(batch.is_prefill)
        self.assertEqual(batch.input_counts, (3,))
        self.assertEqual(batch.state_slots, (0,))
        self.assertEqual((a.num_cached_tokens, a.num_scheduled_tokens), (0, 3))
        self.assertEqual(a.status, SequenceStatus.RUNNING)
        s.postprocess(batch, [44])
        self.assertEqual((a.num_tokens, a.num_cached_tokens, a.num_scheduled_tokens), (4, 3, 0))
        with patch.object(s.admission, 'try_admit', side_effect=AssertionError('must not readmit')):
            batch = s.schedule()
        self.assertFalse(batch.is_prefill)
        s.postprocess(batch, [55])
        self.assertEqual(s.completed[a.seq_id].reason, 'length')
        self.assertEqual(s.completed[a.seq_id].processed_tokens, 4)
        self.assertEqual(a.token_ids, [0, 1, 2, 44, 55])
        self.assertEqual((a.num_cached_tokens, a.num_scheduled_tokens, a.block_table), (0, 0, []))
        self.assertTrue(s.is_finished())
        self.assertIsNone(s.schedule())
        self.assert_resources(s)

    def test_prefill_budget_is_shared_no_chunk(self):
        s = self.make(budget=5)
        a, b, c = self.seq(3), self.seq(3), self.seq(2)
        for r in (a, b, c): s.add(r)
        batch = s.schedule()
        self.assertEqual(batch.sequences, (a, c))  # documented scan-past policy
        self.assertEqual(batch.total_tokens, 5)
        self.assertEqual(list(s.waiting), [b])
        s.postprocess(batch, [10, 20])
        self.assert_resources(s)

    def test_decode_budget_and_rotation(self):
        s = self.make(budget=2, max_seqs=2)
        a, b = self.seq(1, 4), self.seq(1, 4)
        s.add(a); s.add(b)
        s.postprocess(s.schedule(), [10, 20])
        s.token_budget = 1  # shrink per-step budget after prefill
        batch = s.schedule()
        self.assertEqual(batch.total_tokens, 1)
        self.assertEqual(batch.sequences, (a,))
        s.postprocess(batch, [11])
        self.assertEqual(s.schedule().sequences, (b,))

    def test_active_limit_and_waiter_retries_after_finish(self):
        s = self.make(max_seqs=1)
        a, b = self.seq(max_tokens=1), self.seq(max_tokens=1)
        s.add(a); s.add(b)
        batch = s.schedule()
        self.assertEqual(batch.sequences, (a,))
        s.postprocess(batch, [10])
        self.assertEqual(s.schedule().sequences, (b,))
        self.assert_resources(s)

    def test_state_capacity_limits_admission(self):
        s = self.make(slots=1)
        a, b = self.seq(), self.seq()
        s.add(a); s.add(b)
        batch = s.schedule()
        self.assertEqual(batch.sequences, (a,))
        self.assertEqual(list(s.waiting), [b])
        self.assert_resources(s)

    def test_decode_priority_over_new_prefill(self):
        s = self.make()
        a = self.seq()
        s.add(a)
        s.postprocess(s.schedule(), [10])
        b = self.seq()
        s.add(b)
        batch = s.schedule()
        self.assertFalse(batch.is_prefill)
        self.assertEqual(batch.sequences, (a,))
        self.assertEqual(list(s.waiting), [b])

    def test_decode_new_block_retains_same_state_slot(self):
        s = self.make()
        a = self.seq(4)
        s.add(a)
        s.postprocess(s.schedule(), [10])
        slot = s.admission.states.lookup(a.seq_id)
        s.admission.states.recurrent_pool[:, slot].fill_(7)
        batch = s.schedule()
        self.assertEqual(len(a.block_table), 2)
        self.assertEqual(batch.state_slots, (slot,))
        self.assertTrue(torch.all(s.admission.states.recurrent_pool[:, slot] == 7))
        self.assert_resources(s)

    def test_blocked_request_retained_while_neighbor_finishes(self):
        s = self.make(blocks=2)
        a, b = self.seq(4), self.seq(1, 2)
        s.add(a); s.add(b)
        s.postprocess(s.schedule(), [10, 20])
        old = (list(a.block_table), a.num_cached_tokens, s.admission.states.lookup(a.seq_id))
        batch = s.schedule()
        self.assertEqual(batch.sequences, (b,))
        self.assertEqual(a.num_scheduled_tokens, 0)
        self.assertEqual((a.block_table, a.num_cached_tokens, s.admission.states.lookup(a.seq_id)), old)
        s.postprocess(batch, [21])
        self.assertEqual(s.schedule().sequences, (a,))

    def test_no_progress_raises_without_freeing_or_advancing(self):
        s = self.make(blocks=2)
        a, b = self.seq(4), self.seq(4)
        s.add(a); s.add(b)
        s.postprocess(s.schedule(), [10, 20])
        before = [(r.num_cached_tokens, list(r.block_table)) for r in (a, b)]
        with self.assertRaises(NoProgressError): s.schedule()
        self.assertEqual(before, [(r.num_cached_tokens, r.block_table) for r in (a, b)])
        self.assertEqual((a.num_scheduled_tokens, b.num_scheduled_tokens), (0, 0))
        self.assert_resources(s)
        s.fail(a.seq_id, 'explicit capacity policy')
        self.assertEqual(s.schedule().sequences, (b,))

    def test_eos_and_ignore_eos(self):
        s = self.make()
        a, b = self.seq(), self.seq(ignore_eos=True)
        s.add(a); s.add(b)
        s.postprocess(s.schedule(), [99, 99])
        self.assertEqual(s.completed[a.seq_id].reason, 'eos')
        self.assertEqual(s.completed[a.seq_id].processed_tokens, 3)
        self.assertEqual(b.num_cached_tokens, 3)
        self.assertIn(b, s.running)
        self.assert_resources(s)

    def test_zero_generation_skips_all_resources(self):
        s = self.make()
        a = self.seq(max_tokens=0)
        s.add(a)
        self.assertTrue(s.is_finished())
        self.assertEqual(s.completed[a.seq_id].processed_tokens, 0)
        self.assertIsNone(s.schedule())
        self.assert_resources(s)

    def test_invalid_submission_and_duplicate(self):
        s = self.make(budget=4)
        for a in (self.seq(5), self.seq(1, -1)):
            with self.assertRaises(ValueError): s.add(a)
        a = self.seq()
        s.add(a)
        with self.assertRaises(ValueError): s.add(a)
        s.cancel(a.seq_id)
        with self.assertRaises(ValueError): s.add(a)
        self.assert_resources(s)

    def test_unsettled_and_stale_batch_rejected(self):
        s = self.make()
        s.add(self.seq())
        batch = s.schedule()
        with self.assertRaises(RuntimeError): s.schedule()
        s.postprocess(batch, [10])
        with self.assertRaises(ValueError): s.postprocess(batch, [10])
        self.assert_resources(s)

    def test_bad_output_fails_entire_batch_without_partial_append(self):
        for outputs in ([10], [10, -1], [10, 2.5]):
            with self.subTest(outputs=outputs):
                s = self.make()
                a, b = self.seq(), self.seq()
                s.add(a); s.add(b)
                batch = s.schedule()
                with self.assertRaises(ValueError): s.postprocess(batch, outputs)
                self.assertEqual((a.num_tokens, b.num_tokens), (3, 3))
                self.assertTrue(s.is_finished())
                self.assertTrue(all(r.reason == 'failed' for r in s.completed.values()))
                self.assert_resources(s)

    def test_failed_forward_requires_quiescence_then_recycles_dirty_state(self):
        s = self.make(slots=1)
        a = self.seq()
        s.add(a)
        batch = s.schedule()
        s.admission.states.conv_pool.fill_(17)  # simulate partial forward writes
        with self.assertRaises(RuntimeError): s.fail_batch(batch, 'forward error')
        with self.assertRaises(RuntimeError): s.cancel(a.seq_id)
        self.assert_resources(s)
        s.fail_batch(batch, 'forward error', execution_complete=True)
        self.assertEqual(s.completed[a.seq_id].reason, 'failed')
        self.assert_resources(s)
        c = self.seq()
        s.add(c)
        s.schedule()
        self.assertTrue(torch.all(s.admission.states.conv_pool == 0))

    def test_waiting_and_idle_running_failure_isolated(self):
        s = self.make(max_seqs=1)
        a, b = self.seq(), self.seq()
        s.add(a); s.add(b)
        s.postprocess(s.schedule(), [10])
        slot = s.admission.states.lookup(a.seq_id)
        s.admission.states.recurrent_pool[:, slot].fill_(11)
        s.cancel(b.seq_id)
        self.assertTrue(torch.all(s.admission.states.recurrent_pool[:, slot] == 11))
        s.fail(a.seq_id, 'explicit failure')
        self.assertTrue(s.is_finished())
        with self.assertRaises(KeyError): s.fail(a.seq_id)
        self.assert_resources(s)

    def test_admission_exception_fails_only_affected_request(self):
        s = self.make()
        a, b = self.seq(), self.seq()
        s.add(a); s.add(b)
        original = s.admission.states._initialize_slot
        calls = 0
        def init(slot):
            nonlocal calls
            calls += 1
            if calls == 1: raise RuntimeError('injected init failure')
            original(slot)
        with patch.object(s.admission.states, '_initialize_slot', side_effect=init):
            batch = s.schedule()
        self.assertEqual(batch.sequences, (b,))
        self.assertEqual(s.completed[a.seq_id].reason, 'failed')
        self.assert_resources(s)

    def test_decode_allocation_exception_rolls_back_then_recycles(self):
        s = self.make()
        a = self.seq(4)
        s.add(a)
        s.postprocess(s.schedule(), [10])
        original = s.admission.blocks.may_append
        def fail_after_allocation(seq):
            original(seq)
            raise RuntimeError('injected after append')
        with patch.object(s.admission.blocks, 'may_append', side_effect=fail_after_allocation):
            self.assertIsNone(s.schedule())
        self.assertEqual(s.completed[a.seq_id].reason, 'failed')
        self.assert_resources(s)


if __name__ == '__main__':
    unittest.main()
