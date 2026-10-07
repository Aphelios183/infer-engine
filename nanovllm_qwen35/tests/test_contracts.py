import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from qwen35_adapter.contracts import build_layer_maps, parse_text_contract, guard_unimplemented_model


def fixture():
    return {'model_type': 'qwen3_5', 'text_config': {
        'model_type': 'qwen3_5_text', 'num_hidden_layers': 8,
        'layer_types': ['linear_attention'] * 3 + ['full_attention']
                       + ['linear_attention'] * 3 + ['full_attention'],
        'hidden_size': 2560, 'num_attention_heads': 16, 'num_key_value_heads': 4,
        'head_dim': 256, 'max_position_embeddings': 262144,
        'linear_num_key_heads': 16, 'linear_num_value_heads': 32,
        'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
        'linear_conv_kernel_dim': 4, 'dtype': 'bfloat16', 'mamba_ssm_dtype': 'float32',
    }}


class ContractTests(unittest.TestCase):
    def test_student_example(self):
        full, linear = build_layer_maps(fixture()['text_config']['layer_types'])
        self.assertEqual(full, {3: 0, 7: 1})
        self.assertEqual(linear, {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5})

    def test_unknown_type(self):
        with self.assertRaisesRegex(ValueError, "unknown.*layer 1"):
            build_layer_maps(['linear_attention', 'unknown'])

    def test_maps_all_full_or_all_linear(self):
        self.assertEqual(build_layer_maps(['full_attention'] * 2), ({0: 0, 1: 1}, {}))
        self.assertEqual(build_layer_maps(['linear_attention'] * 2), ({}, {0: 0, 1: 1}))

    def test_model_layer_count_must_match(self):
        cfg = fixture()
        cfg['text_config']['num_hidden_layers'] = 9
        with self.assertRaisesRegex(ValueError, 'length'):
            parse_text_contract(cfg)

    def test_layer_types_are_required(self):
        cfg = fixture()
        del cfg['text_config']['layer_types']
        with self.assertRaisesRegex(ValueError, 'explicit list'):
            parse_text_contract(cfg)

    def test_invalid_dimensions(self):
        for value in (0, -1, True, '32'):
            with self.subTest(value=value):
                cfg = fixture()
                cfg['text_config']['num_hidden_layers'] = value
                with self.assertRaises(ValueError):
                    parse_text_contract(cfg)

    def test_unsupported_model_and_missing_text(self):
        for cfg in ({'model_type': 'qwen3'}, {'model_type': 'qwen3_5'},
                    {'model_type': 'qwen3_5_moe'}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                parse_text_contract(cfg)

    def test_head_divisibility(self):
        for field in ('num_key_value_heads', 'linear_num_key_heads'):
            cfg = fixture()
            cfg['text_config'][field] = 3
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_text_contract(cfg)

    def test_shapes_and_dtype_not_silently_assumed(self):
        result = parse_text_contract(fixture())
        self.assertEqual(result['conv_shape_per_layer_per_request'], [8192, 4])
        self.assertEqual(result['recurrent_shape_per_layer_per_request'], [32, 128, 128])
        self.assertEqual(result['declared_ssm_dtype'], 'float32')
        self.assertIn('UNVERIFIED', result['runtime_state_dtype'])
        self.assertFalse(result['native_inference_ready'])

    def test_does_not_mutate_config(self):
        cfg = fixture()
        before = copy.deepcopy(cfg)
        parse_text_contract(cfg)
        self.assertEqual(cfg, before)

    def test_runtime_guard(self):
        for model_type in ('qwen3_5', 'qwen3_5_text', 'qwen3_5_moe'):
            with self.subTest(model_type=model_type), self.assertRaises(NotImplementedError):
                guard_unimplemented_model({'model_type': model_type})
        guard_unimplemented_model({'model_type': 'qwen3'})

    def test_copied_config_rejects_before_auto_config(self):
        # Load this file directly: importing nanovllm would eagerly import GPU dependencies.
        path = Path(__file__).resolve().parents[1] / 'nanovllm' / 'config.py'
        module_name = '_lesson4_nano_config'
        fake_transformers = types.ModuleType('transformers')
        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise AssertionError('AutoConfig must not be reached')
        fake_transformers.AutoConfig = AutoConfig
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'transformers': fake_transformers, module_name: module}):
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                # Generated test fixture, not model weights.
                (Path(tmp) / 'config.json').write_text(json.dumps(fixture()))
                with self.assertRaisesRegex(NotImplementedError, 'native execution'):
                    module.Config(model=tmp)


if __name__ == '__main__':
    unittest.main()
