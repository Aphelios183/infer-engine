from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.state_manager import StateManager
from qwen35_adapter.scheduler import HybridScheduler
from qwen35_adapter.runner_inputs import RunnerInputBuilder


class RunnerInputTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(Sequence, 'block_size', 4)
        p.start()
        self.addCleanup(p.stop)
        self.bm = BlockManager(8, 4)
        self.bm.free_block_ids.clear()
        self.bm.free_block_ids.extend([2, 5, 7, 0, 1, 3, 4, 6])
        self.sm = StateManager(torch.zeros(2, 2, 3, 2), torch.zeros(2, 2, 1, 3, 2))
        self.scheduler = HybridScheduler(
            AdmissionController(self.bm, self.sm, max_prefill_tokens=16),
            max_num_seqs=2, token_budget=16, eos_ids={99})
        self.builder = RunnerInputBuilder(self.scheduler)
        self.b = Sequence([51, 52], SamplingParams(max_tokens=5))
        self.a = Sequence([11, 22, 33], SamplingParams(max_tokens=5))
        self.scheduler.add(self.b)
        self.scheduler.add(self.a)

    def prefill_done(self):
        batch = self.scheduler.schedule()
        self.scheduler.postprocess(batch, [53, 44])

    def test_prefill_packed_order_and_boundaries(self):
        out = self.builder.prepare(self.scheduler.schedule())
        self.assertEqual(out.request_ids, (self.b.seq_id, self.a.seq_id))
        self.assertEqual(out.input_ids.tolist(), [51, 52, 11, 22, 33])
        self.assertEqual(out.positions.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(out.cu_seqlens_q.tolist(), [0, 2, 5])
        self.assertEqual(out.cu_seqlens_k.tolist(), [0, 2, 5])
        self.assertEqual(out.context_lens.tolist(), [2, 3])
        self.assertEqual(out.slot_mapping.tolist(), [8, 9, 20, 21, 22])
        self.assertEqual(out.block_tables.tolist(), [[2], [5]])
        self.assertEqual(out.state_slots.tolist(), [0, 1])
        self.assertEqual(out.last_indices.tolist(), [1, 4])
        self.assertEqual((out.max_seqlen_q, out.max_seqlen_k), (3, 3))

    def test_decode_reordered_requests_keep_state_identity(self):
        self.prefill_done()
        self.scheduler.running.rotate(1)  # control test scheduling order: [A,B]
        out = self.builder.prepare(self.scheduler.schedule())
        self.assertEqual(out.request_ids, (self.a.seq_id, self.b.seq_id))
        self.assertEqual(out.input_ids.tolist(), [44, 53])
        self.assertEqual(out.positions.tolist(), [3, 2])
        self.assertEqual(out.context_lens.tolist(), [4, 3])
        self.assertEqual(out.slot_mapping.tolist(), [23, 10])
        self.assertEqual(out.state_slots.tolist(), [1, 0])
        self.assertEqual(out.cu_seqlens_q.tolist(), [0, 1, 2])
        self.assertEqual(out.cu_seqlens_k.tolist(), [0, 4, 7])
        self.assertEqual(out.last_indices.tolist(), [0, 1])

    def test_decode_cross_block_padding_and_valid_history(self):
        self.prefill_done()
        self.scheduler.running.rotate(1)
        self.scheduler.postprocess(self.scheduler.schedule(), [45, 54])
        out = self.builder.prepare(self.scheduler.schedule())
        self.assertEqual(out.positions.tolist(), [4, 3])
        self.assertEqual(out.slot_mapping.tolist(), [28, 11])
        self.assertEqual(out.block_tables.tolist(), [[5, 7], [2, -1]])
        self.assertEqual(out.context_lens.tolist(), [5, 4])
        history = [out.block_tables[0, p // 4].item() * 4 + p % 4 for p in range(5)]
        self.assertEqual(history, [20, 21, 22, 23, 28])

    def test_prepare_is_read_only_and_tensors_do_not_alias_request(self):
        batch = self.scheduler.schedule()
        before = (self.sm.owners, self.sm.free_slots, list(self.bm.free_block_ids),
                  [(s.num_cached_tokens, s.num_scheduled_tokens, list(s.block_table)) for s in batch.sequences])
        conv = self.sm.conv_pool.clone()
        first = self.builder.prepare(batch)
        first.input_ids[0] = 999
        first.block_tables[0, 0] = 999
        second = self.builder.prepare(batch)
        self.assertEqual(second.input_ids[0].item(), 51)
        self.assertEqual(second.block_tables[0, 0].item(), 2)
        after = (self.sm.owners, self.sm.free_slots, list(self.bm.free_block_ids),
                 [(s.num_cached_tokens, s.num_scheduled_tokens, list(s.block_table)) for s in batch.sequences])
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(conv, self.sm.conv_pool))

    def test_cpu_devices_and_dtypes(self):
        out = self.builder.prepare(self.scheduler.schedule())
        for name, value in vars(out).items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, 'cpu')
                self.assertEqual(value.dtype, torch.int64 if name in ('input_ids', 'positions', 'last_indices') else torch.int32)

    def test_stale_and_forged_batch_rejected(self):
        batch = self.scheduler.schedule()
        with self.assertRaisesRegex(ValueError, 'stale or foreign'):
            self.builder.prepare(replace(batch))
        self.scheduler.postprocess(batch, [53, 44])
        with self.assertRaisesRegex(ValueError, 'stale or foreign'):
            self.builder.prepare(batch)

    def test_corrupt_state_mapping_rejected(self):
        batch = self.scheduler.schedule()
        with patch.object(self.sm, 'lookup', return_value=1):
            with self.assertRaisesRegex(ValueError, 'ownership mismatch'):
                self.builder.prepare(batch)

    def test_missing_block_fails_without_automatic_recycle(self):
        batch = self.scheduler.schedule()
        old = list(self.a.block_table)
        self.a.block_table.clear()  # inject broken metadata
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.builder.prepare(batch)
        self.assertEqual(len(self.sm.owners), 2)
        self.a.block_table[:] = old
        self.scheduler.fail_batch(batch, 'input preparation failed; no forward started', execution_complete=True)
        self.assertTrue(self.scheduler.is_finished())
        self.assertEqual(len(self.bm.free_block_ids), 8)

    def test_invalid_block_id(self):
        batch = self.scheduler.schedule()
        self.a.block_table[0] = 100
        with self.assertRaisesRegex(ValueError, 'physical block'):
            self.builder.prepare(batch)

    def test_duplicate_block_across_requests(self):
        batch = self.scheduler.schedule()
        self.a.block_table[:] = self.b.block_table
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.builder.prepare(batch)

    def test_changed_scheduled_count_rejected(self):
        batch = self.scheduler.schedule()
        self.a.num_scheduled_tokens = 1
        with self.assertRaisesRegex(ValueError, 'scheduled count'):
            self.builder.prepare(batch)

    def test_prefix_or_partial_prefill_not_silently_enabled(self):
        batch = self.scheduler.schedule()
        self.a.num_cached_tokens = 1
        with self.assertRaises(ValueError):
            self.builder.prepare(batch)

    def test_real_scheduler_builder_completion_loop(self):
        steps = 0
        while not self.scheduler.is_finished():
            batch = self.scheduler.schedule()
            out = self.builder.prepare(batch)
            self.assertEqual(out.input_ids.numel(), batch.total_tokens)
            self.assertEqual(out.state_slots.numel(), len(batch.sequences))
            self.scheduler.postprocess(batch, [99] * len(batch.sequences))  # synthetic EOS, not inference
            steps += 1
        self.assertEqual(steps, 1)
        self.assertEqual(self.sm.owners, {})
        self.assertEqual(len(self.bm.free_block_ids), 8)


if __name__ == '__main__':
    unittest.main()
