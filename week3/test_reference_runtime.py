"""CPU tests: input contracts, positions, masks, output safety and load errors."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch

try:
    from week3 import reference_runtime as runtime
except ImportError:
    runtime = None

class CaptureModel:
    def __init__(self):
        self.config = NS(text_config=NS(vocab_size=8), image_token_id=6, video_token_id=5)
        self.kwargs = None
        self.bad = None
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        logits = torch.zeros(1, 1, 8)
        if self.bad == "nan":
            logits[0, 0, 0] = float("nan")
        if self.bad == "shape":
            logits = torch.zeros(2, 1, 8)
        cache = NS(get_seq_length=lambda: kwargs["attention_mask"].shape[1])
        if self.bad == "cache":
            cache = None
        return NS(logits=logits, past_key_values=cache if kwargs["use_cache"] else None)

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, "reference_runtime is not implemented")
        self.model = CaptureModel()
    def test_decode_positions_mask_and_cache(self):
        cache = NS(get_seq_length=lambda: 7)
        logits, state = runtime.forward_last(self.model, torch.tensor([[3]]), processed_tokens=7, cache=cache)
        self.assertEqual(tuple(logits.shape), (1, 8))
        self.assertEqual(self.model.kwargs["position_ids"].tolist(), [[[7]], [[7]], [[7]], [[7]]])
        self.assertEqual(self.model.kwargs["attention_mask"].tolist(), [[1]*8])
        self.assertIs(self.model.kwargs["past_key_values"], cache)
        self.assertEqual(state.get_seq_length(), 8)
        self.assertNotIn("pixel_values", self.model.kwargs)
    def test_prefill_positions_and_full_recompute(self):
        _, cache = runtime.forward_last(self.model, torch.tensor([[1,2,3]]), use_cache=False)
        self.assertEqual(self.model.kwargs["position_ids"].tolist(), [[[0,1,2]]]*4)
        self.assertIsNone(cache)
    def test_reject_bad_inputs_before_forward(self):
        for ids in [torch.empty((1,0),dtype=torch.long), torch.ones((2,1),dtype=torch.long),
                    torch.tensor([[1.0]]), torch.tensor([[-1]]), torch.tensor([[8]]),
                    torch.tensor([[6]]), torch.tensor([[5]])]:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    runtime.forward_last(self.model, ids)
                self.assertIsNone(self.model.kwargs)
    def test_reject_invalid_cache_position_contract(self):
        ids = torch.tensor([[1]])
        for kw in [dict(processed_tokens=-1),dict(processed_tokens=True),
                   dict(processed_tokens=2),dict(cache=NS(get_seq_length=lambda: 1)),
                   dict(processed_tokens=2,cache=NS(get_seq_length=lambda: 1)),
                   dict(use_cache=False,cache=object()),dict(use_cache=False,processed_tokens=2)]:
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    runtime.forward_last(self.model, ids, **kw)
    def test_reject_bad_model_outputs(self):
        for bad in ["nan","shape","cache"]:
            with self.subTest(bad=bad):
                self.model.bad = bad
                with self.assertRaises(ValueError):
                    runtime.forward_last(self.model, torch.tensor([[1]]))
    def test_eos_normalization(self):
        self.assertEqual(runtime.normalize_eos([2,7],8), {2,7})
        self.assertEqual(runtime.normalize_eos(7,8), {7})
        for bad in [None,[],True,-1,8,[2,False],[1.5]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime.normalize_eos(bad,8)
    def test_loading_diagnostics_are_not_ignored(self):
        for key in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.check_loading_info({key:["broken"]})
        runtime.check_loading_info({})
    def test_unknown_model_directory_is_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                runtime.verify_local_snapshot(Path(folder))

    def test_extra_loader_asset_cannot_bypass_snapshot_check(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root/"config.json").write_text(json.dumps({"model_type":"qwen3_5"}))
            (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"x":"part.safetensors"}}))
            (root/"part.safetensors").write_bytes(b"fake verified shard")
            tracked = ["config.json","model.safetensors.index.json","part.safetensors"]
            def fake_git(args, **kwargs):
                command = args[3:]
                outputs = {
                    ("rev-parse","--show-toplevel"): str(root),
                    ("rev-parse","HEAD"): runtime.MODEL_REVISION,
                    ("remote","get-url","origin"): runtime.MODEL_ORIGIN,
                    ("status","--porcelain","--untracked-files=no"): "",
                    ("ls-files","-z"): "\0".join(tracked)+"\0",
                    ("lfs","fsck"): "Git LFS fsck OK",
                }
                return NS(stdout=outputs[tuple(command)])
            with patch.object(runtime.subprocess,"run",side_effect=fake_git):
                self.assertEqual(runtime.verify_local_snapshot(root)["revision"],runtime.MODEL_REVISION)
                # Both untracked and gitignored assets must be rejected by directory inventory.
                for name in ["model.safetensors","generation_config.json"]:
                    with self.subTest(name=name):
                        (root/name).write_bytes(b"untracked override")
                        with self.assertRaisesRegex(ValueError,"未跟踪|untracked"):
                            runtime.verify_local_snapshot(root)
                        (root/name).unlink()

if __name__ == "__main__":
    unittest.main()
