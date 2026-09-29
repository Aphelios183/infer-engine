"""Verifier guardrails: no GPU/model required; real integration runs separately."""
import importlib
import unittest
from unittest.mock import patch
import torch


class VerificationTests(unittest.TestCase):
    def test_isolation_exercises_decode_error_and_same_generation_entry(self):
        def step(model, ids, *, processed_tokens=0, cache=None, use_cache=True):
            total = int(ids.sum()) + (0 if cache is None else cache['sum'])
            return torch.tensor([[float(total), 0.]]), {'sum': total}
        greedy = self.v.run_greedy
        def generation(*a, **kw):
            kw.setdefault('step', step)
            return greedy(*a, **kw)
        with patch.object(self.v, 'forward_last', step), patch.object(self.v, 'run_greedy', generation):
            r = self.v.verify_isolation(None, torch.tensor([[1,2]]), torch.tensor([[3]]), torch.tensor([[4,5]]), {1}, limits=dict(max_abs=0., mean_abs=0., relative_l2=0.))
        failure = r['cases'][1]
        self.assertEqual(failure['fault']['phase'], 'decode')
        self.assertTrue(failure['fault']['received_existing_cache'])
        self.assertEqual(failure['fault']['position_start'], 2)
        self.assertEqual(len(failure['generation_steps']), 4)
        self.assertTrue(all(x['status']=='pass' for x in failure['generation_steps']))
        self.assertEqual(r['status'], 'pass')

    def setUp(self):
        try:
            self.v = importlib.import_module('week3.verify_generation')
        except ModuleNotFoundError:
            self.fail('verification module not implemented')

    def test_metrics_and_top1_disagreement(self):
        m = self.v.compare_logits(torch.tensor([[1., .999]]), torch.tensor([[.999, 1.]]))
        self.assertAlmostEqual(m['max_abs'], .001, places=6)
        self.assertNotEqual(m['top1_reference'], m['top1_candidate'])
        self.assertEqual(self.v.evaluate_metrics(m, None), 'diagnostic')
        self.assertEqual(self.v.evaluate_metrics(m, dict(max_abs=.01, mean_abs=.01, relative_l2=.01)), 'needs_review')

    def test_matching_argmax_does_not_hide_error(self):
        m = self.v.compare_logits(torch.tensor([[10., 2., 1.]]), torch.tensor([[10., 2., 5.]]))
        self.assertEqual(m['max_abs'], 4.)
        self.assertEqual(self.v.evaluate_metrics(m, dict(max_abs=.01, mean_abs=.01, relative_l2=.01)), 'fail')

    def test_zero_and_invalid_logits(self):
        z = torch.zeros(1, 3)
        self.assertEqual(self.v.compare_logits(z, z)['relative_l2'], 0.)
        m = self.v.compare_logits(z, torch.ones_like(z))
        self.assertIsNone(m['relative_l2'])
        self.assertEqual(self.v.evaluate_metrics(m, dict(max_abs=9., mean_abs=9., relative_l2=9.)), 'fail')
        for x in [torch.ones(1, 2), torch.empty(1, 0), torch.tensor([[float('nan'), 0., 0.]]), torch.tensor([[float('inf'), 0., 0.]])]:
            with self.assertRaises(ValueError):
                self.v.compare_logits(z, x)

    def test_tolerance_requires_approval_and_matching_fingerprint(self):
        fp = {'revision': 'a', 'dtype': 'bf16'}
        doc = dict(approved=False, fingerprint=fp.copy(), review_reason='independent calibration', limits=dict(max_abs=.1, mean_abs=.1, relative_l2=.1))
        with self.assertRaises(ValueError):
            self.v.validate_tolerances(doc, fp)
        doc['approved'] = True
        self.assertEqual(self.v.validate_tolerances(doc, fp), doc['limits'])
        with self.assertRaises(ValueError):
            self.v.validate_tolerances(doc, {'revision': 'b', 'dtype': 'bf16'})
        for invalid in [float('nan'), float('inf'), -1., True]:
            doc['limits']['max_abs'] = invalid
            with self.assertRaises(ValueError):
                self.v.validate_tolerances(doc, fp)

    def test_fixed_prefix_inputs_positions_and_error_detection(self):
        calls = []
        def step(model, ids, *, processed_tokens=0, cache=None, use_cache=True):
            calls.append((ids.tolist()[0], processed_tokens, cache is None, use_cache))
            total = ids.sum().item() + (0 if cache is None else cache['sum'])
            return torch.tensor([[float(total), 0.]]), {'sum': total} if use_cache else None
        args = (None, torch.tensor([[1, 2]]), torch.tensor([[3, 4]]))
        limits = dict(max_abs=0., mean_abs=0., relative_l2=0.)
        result = self.v.verify_prefixes(*args, limits=limits, step=step)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(len(result['steps']), 3)
        self.assertEqual(calls, [([1,2],0,True,False),([1,2],0,True,True),([1,2,3],0,True,False),([3],2,False,True),([1,2,3,4],0,True,False),([4],3,False,True)])
        def bad(*a, **kw):
            if kw.get('cache') is not None:
                kw['cache']['sum'] += 1
            return step(*a, **kw)
        result = self.v.verify_prefixes(*args, limits=limits, step=bad)
        self.assertEqual(result['steps'][0]['status'], 'pass')
        self.assertEqual(result['steps'][1]['status'], 'fail')
        self.assertEqual(result['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
