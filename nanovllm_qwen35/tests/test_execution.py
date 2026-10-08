"""Synthetic CPU backend tests, NOT Qwen3.5 numerical validation."""
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.state_manager import StateManager
from qwen35_adapter.scheduler import HybridScheduler
from qwen35_adapter.execution import CPUExecutionRunner


class ToyBackend:
    execution_mode = 'cpu_sync'

    def __init__(self, states):
        self.states = states
        self.kv = torch.zeros(32)  # one scalar/token: NOT a real KV layout
        self.calls = []
        self.bad = None

    def forward(self, inputs):
        self.calls.append(('forward', inputs))
        self.kv[inputs.slot_mapping.long()] = inputs.input_ids.float()
        for slot in inputs.state_slots.tolist():
            self.states.conv_pool[:, slot].add_(1)
            self.states.recurrent_pool[:, slot].add_(1)
        if self.bad == 'forward':
            raise RuntimeError('partial state update')
        return inputs.input_ids.float().reshape(-1, 1)

    def compute_logits(self, hidden):
        self.calls.append(('logits', hidden.clone()))
        logits = torch.zeros(len(hidden), 100)
        logits.scatter_(1, hidden.long() + 1, 10)
        if self.bad == 'nan':
            logits[0, 0] = float('nan')
        if self.bad == 'rows':
            logits = logits[:1]
        return logits


class ToySampler:
    execution_mode = 'cpu_sync'

    def sample(self, logits, temperatures):
        self.temperatures = temperatures
        return logits.argmax(-1)  # deliberately greedy fixture, not nano sampler


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(Sequence, 'block_size', 4)
        p.start()
        self.addCleanup(p.stop)
        self.sm = StateManager(torch.zeros(2, 2, 3, 2), torch.zeros(2, 2, 1, 3, 2))
        self.bm = BlockManager(8, 4)
        self.scheduler = HybridScheduler(
            AdmissionController(self.bm, self.sm, max_prefill_tokens=16),
            max_num_seqs=2, token_budget=16, eos_ids={99})
        self.b = Sequence([51, 52], SamplingParams(max_tokens=2, temperature=0.7))
        self.a = Sequence([11, 22, 33], SamplingParams(max_tokens=2, temperature=0.9))
        for seq in (self.b, self.a):
            self.scheduler.add(seq)
        self.backend = ToyBackend(self.sm)
        self.sampler = ToySampler()
        self.runner = CPUExecutionRunner(self.scheduler, self.backend, self.sampler)

    def assert_released(self):
        self.assertTrue(self.scheduler.is_finished())
        self.assertEqual(self.sm.owners, {})
        self.assertEqual(len(self.bm.free_block_ids), 8)

    def test_prefill_decode_and_last_hidden_selection(self):
        first = self.runner.run(self.scheduler.schedule())
        self.assertEqual(first.token_ids, (53, 34))
        self.assertEqual(self.backend.calls[1][1].flatten().tolist(), [52, 33])
        self.assertEqual(self.sampler.temperatures, (0.7, 0.9))
        self.assertEqual((self.b.num_cached_tokens, self.a.num_cached_tokens), (2, 3))
        self.assertTrue(torch.all(self.sm.conv_pool == 1))
        self.scheduler.running.rotate(1)
        second = self.runner.run(self.scheduler.schedule())
        self.assertEqual(second.request_ids, (self.a.seq_id, self.b.seq_id))
        self.assertEqual(second.token_ids, (35, 54))
        self.assertEqual(self.backend.calls[2][1].state_slots.tolist(), [1, 0])
        self.assertTrue(torch.all(self.sm.conv_pool == 2))  # no per-step reset
        self.assert_released()
        self.assertEqual(self.scheduler.completed[self.a.seq_id].processed_tokens, 4)

    def test_eos_not_forwarded(self):
        with patch.object(self.sampler, 'sample', return_value=torch.tensor([99, 99])):
            result = self.runner.run(self.scheduler.schedule())
        self.assertEqual(result.token_ids, (99, 99))
        self.assertNotIn(99, self.backend.kv.tolist())
        self.assert_released()

    def test_partial_forward_failure_disposes_not_rolls_back(self):
        self.backend.bad = 'forward'
        with self.assertRaisesRegex(RuntimeError, 'partial'):
            self.runner.run(self.scheduler.schedule())
        self.assert_released()
        self.assertTrue(torch.all(self.sm.recurrent_pool == 1))
        self.assertEqual(self.a.token_ids, [11, 22, 33])
        self.assertEqual(self.scheduler.completed[self.a.seq_id].reason, 'failed')

    def test_bad_logits_fail_whole_batch(self):
        self.backend.bad = 'nan'
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            self.runner.run(self.scheduler.schedule())
        self.assert_released()

    def test_wrong_logits_rows(self):
        self.backend.bad = 'rows'
        with self.assertRaisesRegex(ValueError, 'matrix'):
            self.runner.run(self.scheduler.schedule())
        self.assert_released()

    def test_sampler_exception(self):
        with patch.object(self.sampler, 'sample', side_effect=RuntimeError('sampling failed')):
            with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
                self.runner.run(self.scheduler.schedule())
        self.assert_released()

    def test_out_of_vocab_rejected(self):
        with patch.object(self.sampler, 'sample', return_value=torch.tensor([100, 1])):
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                self.runner.run(self.scheduler.schedule())
        self.assert_released()

    def test_sampler_shape_rejected(self):
        with patch.object(self.sampler, 'sample', return_value=torch.tensor([[1], [2]])):
            with self.assertRaisesRegex(ValueError, 'int64'):
                self.runner.run(self.scheduler.schedule())
        self.assert_released()

    def test_stale_batch_cannot_execute_twice(self):
        batch = self.scheduler.schedule()
        self.runner.run(batch)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.runner.run(batch)
        self.assertEqual(len(self.backend.calls), 2)

    def test_prepare_failure_keeps_resources(self):
        batch = self.scheduler.schedule()
        with patch.object(self.runner.builder, 'prepare', side_effect=ValueError('bad metadata')):
            with self.assertRaisesRegex(ValueError, 'metadata'):
                self.runner.run(batch)
        self.assertIs(self.scheduler._pending, batch)
        self.assertEqual(len(self.sm.owners), 2)
        self.assertEqual(self.backend.calls, [])

    def test_non_cpu_sync_backend_rejected(self):
        self.backend.execution_mode = 'cuda_async'
        with self.assertRaisesRegex(ValueError, 'cpu_sync'):
            CPUExecutionRunner(self.scheduler, self.backend, self.sampler)


if __name__ == '__main__':
    unittest.main()
