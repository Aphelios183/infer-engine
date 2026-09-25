"""CPU/offline: real model config arithmetic and chat-template encoding."""
import importlib.util
import unittest


class InspectTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('week3.inspect_model'),
                             'inspect_model not implemented')
        from week3 import inspect_model
        self.module = inspect_model
        self.config = dict(num_hidden_layers=28, num_attention_heads=16,
                           num_key_value_heads=8, head_dim=128, hidden_size=1024,
                           vocab_size=151936)

    def test_explicit_head_dim_controls_projection_and_kv_bytes(self):
        result = self.module.describe_config(self.config, batch=1, length=4096)
        self.assertEqual(result['q_projection_width'], 2048)
        self.assertEqual(result['kv_projection_width'], 1024)
        self.assertEqual(result['q_heads_per_kv'], 2)
        self.assertEqual(result['kv_bytes'], 469762048)
        self.assertEqual(result['kv_mib'], 448)

    def test_batch_and_length_scale_cache_not_query_heads(self):
        result = self.module.describe_config(self.config, batch=2, length=2048)
        self.assertEqual(result['kv_mib'], 448)

    def test_invalid_dimensions_rejected(self):
        for batch,length in [(0,4096),(1,-1)]:
            with self.assertRaises(ValueError):
                self.module.describe_config(self.config,batch=batch,length=length)

    def test_local_template_matches_known_ids_and_roundtrip(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            '/home/ubuntu/huggingface/Qwen3-0.6B', local_files_only=True,
            trust_remote_code=False)
        result = self.module.inspect_prompt(tokenizer, '用一句话解释 KV Cache。')
        self.assertEqual(result['chat_ids'], [151644,872,198,11622,105321,
                          104136,84648,19479,1773,151645,198,151644,77091,
                          198,151667,271,151668,271])
        self.assertEqual(result['chat_length'],18)
        self.assertTrue(result['template_encoding_matches'])
        self.assertTrue(result['roundtrip_matches'])
        self.assertEqual(result['eos_token_id'],151645)


if __name__ == '__main__':
    unittest.main(verbosity=2)
