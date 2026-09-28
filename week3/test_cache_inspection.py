"""CPU checks for read-only hybrid cache metadata; no model or CUDA needed."""
import json
import unittest
from types import SimpleNamespace as NS
import torch
try:
    from week3.cache_inspection import snapshot_cache
except ImportError:
    snapshot_cache = None

class CacheInspectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(snapshot_cache, "snapshot_cache not implemented")
    def fixture(self):
        base=torch.zeros(1,1,2,2,dtype=torch.bfloat16)
        return NS(layers=[
            NS(keys=base,values=base.view_as(base)),
            NS(conv_states=torch.zeros(1,2,4,dtype=torch.bfloat16),
               recurrent_states=torch.zeros(1,1,2,2,dtype=torch.float32))])
    def test_shared_storage_actual_dtype_and_no_mutation(self):
        cache=self.fixture()
        before=[t.clone() for layer in cache.layers for t in vars(layer).values()]
        r=snapshot_cache(cache,["full_attention","linear_attention"],2)
        self.assertEqual(r["logical_tensor_bytes"],48)
        self.assertEqual(r["unique_storage_bytes"],40)
        self.assertEqual(r["layers"][1]["tensors"]["recurrent_states"]["dtype"],"torch.float32")
        self.assertEqual(r["layers"][0]["effective_length"],2)
        self.assertTrue(r["complete"])
        after=[t for layer in cache.layers for t in vars(layer).values()]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        json.dumps(r,allow_nan=False)
    def test_dtype_is_observed_not_configured(self):
        cache=self.fixture()
        cache.layers[1].recurrent_states=cache.layers[1].recurrent_states.bfloat16()
        r=snapshot_cache(cache,["full_attention","linear_attention"],2)
        self.assertEqual((r["logical_tensor_bytes"],r["unique_storage_bytes"]),(40,32))
    def test_offset_views_deduplicate_whole_storage(self):
        base=torch.zeros(16,dtype=torch.float32)
        cache=NS(layers=[NS(keys=base[:4].view(1,1,2,2),values=base[4:8].view(1,1,2,2))])
        r=snapshot_cache(cache,["full_attention"],2)
        self.assertEqual((r["logical_tensor_bytes"],r["unique_storage_bytes"]),(32,64))
    def test_each_call_has_independent_storage_accounting(self):
        cache=self.fixture()
        a=snapshot_cache(cache,["full_attention","linear_attention"],2)
        b=snapshot_cache(cache,["full_attention","linear_attention"],2)
        self.assertEqual(a,b)
    def test_missing_cache_is_unknown_not_zero(self):
        r=snapshot_cache(None,["full_attention","linear_attention"],0)
        self.assertFalse(r["complete"])
        self.assertIsNone(r["logical_tensor_bytes"])
        self.assertIsNone(r["unique_storage_bytes"])
        self.assertIsNone(r["layers"][0]["tensors"]["keys"])
    def test_missing_attribute_and_none_are_incomplete(self):
        for layer in [NS(keys=None,values=None),NS(keys=torch.zeros(1,1,2,2))]:
            with self.subTest(layer=layer):
                r=snapshot_cache(NS(layers=[layer]),["full_attention"],2)
                self.assertFalse(r["complete"])
                self.assertIsNone(r["logical_tensor_bytes"])
                self.assertIsNone(r["layers"][0]["tensors"]["values"])
    def test_allocated_zero_tensor_is_not_unknown(self):
        t=torch.empty(1,1,0,2)
        r=snapshot_cache(NS(layers=[NS(keys=t,values=t)]),["full_attention"],0)
        self.assertTrue(r["complete"])
        self.assertEqual(r["logical_tensor_bytes"],0)
    def test_bad_schema_and_length_are_rejected(self):
        cache=self.fixture()
        for types,n in [(["unknown","linear_attention"],2),(["full_attention"],2),
                        (["full_attention","linear_attention"],3),
                        (["full_attention","linear_attention"],True)]:
            with self.subTest(types=types,n=n), self.assertRaises(ValueError):
                snapshot_cache(cache,types,n)
        with self.assertRaises(ValueError):
            snapshot_cache(NS(),["full_attention"],2)
    def test_non_tensor_state_is_rejected(self):
        with self.assertRaises(ValueError):
            snapshot_cache(NS(layers=[NS(conv_states="bad",recurrent_states=None)]),
                           ["linear_attention"],1)
    def test_mismatching_kv_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            snapshot_cache(NS(layers=[NS(keys=torch.zeros(1,1,2,2),values=torch.zeros(1,2,2,2))]),
                           ["full_attention"],2)

if __name__=="__main__":
    unittest.main()
