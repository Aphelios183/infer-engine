"""These tests catch extra decode calls, lost cache, and incorrect stop conditions."""
import json
import unittest
from types import SimpleNamespace as NS
import torch
try:
    from week3 import minimal_generate as generation
except ImportError:
    generation = None

class ScriptedStep:
    def __init__(self, tokens=(3,4,7), fail_at=None):
        self.tokens = tokens
        self.calls = []
        self.fail_at = fail_at
        self.marker = object()
    def __call__(self, model, ids, *, processed_tokens=0, cache=None, use_cache=True):
        self.calls.append((ids.tolist(), processed_tokens, cache))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated forward failure")
        scores = torch.full((1,8), -10.0)
        scores[0,self.tokens[(len(self.calls)-1) % len(self.tokens)]] = 10.
        return scores, self.marker

class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(generation, "minimal_generate is not implemented")
        self.ids = torch.tensor([[7,2]])
    def test_zero_budget_does_not_call_forward(self):
        step = ScriptedStep()
        r = generation.run_greedy(None,self.ids,{7},0,step=step)
        self.assertEqual((r["output_ids"],r["processed_tokens"],r["trace"]), ([],0,[]))
        self.assertEqual(step.calls,[])
    def test_one_budget_prefill_only(self):
        step = ScriptedStep()
        r = generation.run_greedy(None,self.ids,{7},1,step=step)
        self.assertEqual(r["output_ids"],[3])
        self.assertEqual(r["processed_tokens"],2)
        self.assertEqual(len(step.calls),1)
    def test_two_budget_no_third_forward(self):
        step = ScriptedStep()
        r = generation.run_greedy(None,self.ids,{7},2,step=step)
        self.assertEqual(r["output_ids"],[3,4])
        self.assertEqual(r["processed_tokens"],3)
        self.assertEqual(r["stop_reason"],"length")
        self.assertEqual(len(step.calls),2)
        self.assertEqual(step.calls[1][:2],([[3]],2))
        self.assertIs(step.calls[1][2],step.marker)
        self.assertEqual([e["phase"] for e in r["trace"]],["prefill","decode"])
        self.assertEqual(r["trace"][-1]["stop_reason"],"length")
        self.assertEqual(r["trace"][-1]["position_start"],2)
        json.dumps(r,allow_nan=False)
    def test_eos_in_prompt_does_not_stop_but_new_eos_does(self):
        step = ScriptedStep((3,7))
        r = generation.run_greedy(None,self.ids,[7],8,step=step)
        self.assertEqual(r["output_ids"],[3,7])
        self.assertEqual(r["processed_tokens"],3)
        self.assertEqual(r["stop_reason"],"eos")
        self.assertEqual(len(step.calls),2)
    def test_first_token_eos_and_eos_priority(self):
        step=ScriptedStep((7,))
        r=generation.run_greedy(None,self.ids,{7},1,step=step)
        self.assertEqual(r["stop_reason"],"eos")
        self.assertEqual(r["processed_tokens"],2)
        self.assertEqual(r["output_ids"],[7])
    def test_invalid_budget(self):
        for value in [-1,True,1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation.run_greedy(None,self.ids,{7},value,step=ScriptedStep())
    def test_invalid_input(self):
        for ids in [torch.empty((1,0),dtype=torch.long),torch.ones((2,1),dtype=torch.long),
                    torch.tensor([[1.]]),torch.tensor([[-1]])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                generation.run_greedy(None,ids,{7},2,step=ScriptedStep())
    def test_nonfinite_and_bad_shape_scores(self):
        for scores in [torch.tensor([[float("nan")]*8]),torch.zeros(2,8),torch.zeros(1,1,8)]:
            def step(*args,**kwargs):
                return scores, object()
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                generation.run_greedy(None,self.ids,{7},2,step=step)
    def test_missing_cache_rejected(self):
        def step(*args,**kwargs):
            return torch.zeros(1,8),None
        with self.assertRaises(ValueError):
            generation.run_greedy(None,self.ids,{7},2,step=step)
    def test_failure_then_new_request_starts_clean(self):
        step=ScriptedStep(fail_at=2)
        with self.assertRaisesRegex(RuntimeError,"simulated"):
            generation.run_greedy(None,self.ids,{7},3,step=step)
        fresh=ScriptedStep()
        generation.run_greedy(None,self.ids,{7},1,step=fresh)
        self.assertEqual(fresh.calls[0][:2],([[7,2]],0))
        self.assertIsNone(fresh.calls[0][2])
    def test_invalid_eos(self):
        for eos in [set(),[True],[-1],[8]]:
            with self.subTest(eos=eos), self.assertRaises(ValueError):
                generation.run_greedy(None,self.ids,eos,1,step=ScriptedStep())
    def test_observer_must_return_json_not_cache_tensor(self):
        with self.assertRaises((TypeError,ValueError)):
            generation.run_greedy(None,self.ids,{7},1,step=ScriptedStep(),
                                  observe_cache=lambda c,n: {"tensor":torch.zeros(1)})

if __name__ == "__main__":
    unittest.main()
