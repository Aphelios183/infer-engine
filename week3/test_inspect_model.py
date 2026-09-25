"""CPU/offline: real model config arithmetic and chat-template encoding."""
import importlib.util
import copy
import json
from pathlib import Path
import unittest


class InspectTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('week3.inspect_model'),
                             'inspect_model not implemented')
        from week3 import inspect_model
        self.module = inspect_model
        self.config = dict(model_type='qwen3', num_hidden_layers=28, num_attention_heads=16,
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


class HybridConfigTest(unittest.TestCase):
    def setUp(self):
        from week3.inspect_model import describe_config
        self.describe = describe_config
        self.config = json.loads((Path(__file__).parent / 'fixtures' /
            'qwen35_4b_config_minimal.json').read_text(encoding='utf-8'))

    def inspect(self, config=None, **kwargs):
        try:
            return self.describe(self.config if config is None else config, **kwargs)
        except Exception as exc:
            self.fail(f'合法配置应成功读取，实际为 {type(exc).__name__}: {exc}')

    def assert_invalid(self, config, field, **kwargs):
        try:
            self.describe(config, **kwargs)
        except ValueError as exc:
            self.assertIn(field, str(exc))
        except Exception as exc:
            self.fail(f'应明确拒绝字段 {field}，实际为 {type(exc).__name__}: {exc}')
        else:
            self.fail(f'非法字段 {field} 未被拒绝')

    def test_only_full_attention_layers_contribute_kv_bytes(self):
        r = self.inspect()
        self.assertEqual((r['full_attention_layers'], r['linear_attention_layers']), (8, 24))
        self.assertEqual((r['kv_bytes'], r['kv_mib']), (134217728, 128))
        self.assertEqual(r['kv_scope'], 'full_attention_only')
        self.assertIsNone(r['linear_state_bytes'])
        self.assertEqual(r['full_attention_layer_indices'], [3, 7, 11, 15, 19, 23, 27, 31])

    def test_explicit_head_dim_and_gate_do_not_inflate_query_features(self):
        r = self.inspect()
        self.assertEqual(r['q_projection_width'], 4096)
        self.assertEqual(r['kv_projection_width'], 1024)
        self.assertEqual(r['q_heads_per_kv'], 4)
        self.assertEqual(r['q_projection_scope'], 'query_features_only')

    def test_batch_length_and_element_size_scale_only_kv(self):
        self.assertEqual(self.inspect(batch=2)['kv_bytes'], 268435456)
        self.assertEqual(self.inspect(length=2048)['kv_mib'], 64)
        self.assertEqual(self.inspect(bytes_per_element=4)['kv_mib'], 256)

    def test_unknown_and_missing_model_types_are_rejected(self):
        for cfg in ({}, {'model_type': 'unknown'}, [], None):
            self.assert_invalid(cfg, 'model_type')

    def test_nested_config_must_be_a_text_config(self):
        for text in (None, [], {}, {'model_type': 'qwen3'}):
            self.assert_invalid({'model_type': 'qwen3_5', 'text_config': text}, 'text_config')

    def test_layer_list_must_match_count_and_known_types(self):
        for value in (None, 'full_attention', ['full_attention'], ['bad'] * 32):
            cfg = copy.deepcopy(self.config)
            cfg['text_config']['layer_types'] = value
            self.assert_invalid(cfg, 'layer_types')

    def test_missing_and_invalid_dimensions_name_the_field(self):
        fields = ['num_hidden_layers', 'hidden_size', 'num_attention_heads',
                  'num_key_value_heads', 'head_dim', 'vocab_size',
                  'linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim']
        for field in fields:
            for value in (None, True, 0, -1, 1.5, '4'):
                with self.subTest(field=field, value=value):
                    cfg = copy.deepcopy(self.config)
                    cfg['text_config'][field] = value
                    self.assert_invalid(cfg, field)
            cfg = copy.deepcopy(self.config)
            del cfg['text_config'][field]
            self.assert_invalid(cfg, field)

    def test_runtime_sizes_reject_bool_fractional_and_nonpositive(self):
        for field in ('batch', 'length', 'bytes_per_element'):
            for value in (True, 0, -1, 1.5):
                self.assert_invalid(self.config, field, **{field: value})

    def test_invalid_gqa_ratio_is_rejected(self):
        cfg = copy.deepcopy(self.config)
        cfg['text_config']['num_attention_heads'] = 15
        self.assert_invalid(cfg, 'num_attention_heads')

    def test_describe_does_not_mutate_input(self):
        original = copy.deepcopy(self.config)
        r = self.inspect()
        r['layer_types'].clear()
        self.assertEqual(self.config, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
