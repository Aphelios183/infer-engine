"""CPU/offline: real model config arithmetic and chat-template encoding."""
import importlib.util
import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
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


class TinyTokenizer:
    """No model assets: a reversible codec with explicit chat-template branches."""
    eos_token, eos_token_id, vocab_size = '<eos>', 9, 1114112
    mapping = True
    ignore_switch = False
    mismatch = False
    bad_roundtrip = False

    def get_chat_template(self):
        return '{% if enable_thinking %}[on]{% else %}[off]{% endif %}'

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = '[user]' + messages[0]['content'] + '[assistant]'
        text += '[on]' if enable_thinking and not self.ignore_switch else '[off]'
        if not tokenize:
            return text
        ids = [ord(c) for c in text] + ([0] if self.mismatch else [])
        return {'input_ids': ids} if self.mapping else ids

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text] + ([0] if add_special_tokens else [])

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids) + ('!' if self.bad_roundtrip else '')

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]

    def __len__(self):
        return self.vocab_size


class PromptTest(unittest.TestCase):
    def inspect(self, tokenizer, text, **kwargs):
        from week3.inspect_model import inspect_prompt
        return inspect_prompt(tokenizer, text, **kwargs)

    def test_mapping_and_list_encode_template_once(self):
        for mapping in (True, False):
            tok = TinyTokenizer()
            tok.mapping = mapping
            r = self.inspect(tok, '你好')
            self.assertEqual(r['rendered'], '[user]你好[assistant][off]')
            self.assertEqual(r['raw_ids'], [20320, 22909])
            self.assertEqual(r['chat_ids'], [ord(c) for c in '[user]你好[assistant][off]'])
            self.assertTrue(r['thinking_switch_verified'])
            self.assertEqual(r['template_settings']['enable_thinking'], False)
            self.assertEqual(len(r['template_sha256']), 64)

    def test_thinking_on_is_recorded_and_changes_rendering(self):
        r = self.inspect(TinyTokenizer(), 'hello', thinking=True)
        self.assertTrue(r['template_settings']['enable_thinking'])
        self.assertTrue(r['rendered'].endswith('[on]'))

    def test_empty_nontext_and_nonboolean_inputs_rejected(self):
        for text in ('', ' \n ', None, ['hello']):
            with self.assertRaises(ValueError):
                self.inspect(TinyTokenizer(), text)
        with self.assertRaises(ValueError):
            self.inspect(TinyTokenizer(), 'hello', thinking='off')

    def test_ignored_thinking_switch_is_not_claimed_as_supported(self):
        tok = TinyTokenizer()
        tok.ignore_switch = True
        with self.assertRaisesRegex(ValueError, 'thinking'):
            self.inspect(tok, 'hello')

    def test_different_template_encodings_are_rejected(self):
        tok = TinyTokenizer()
        tok.mismatch = True
        with self.assertRaises(AssertionError):
            self.inspect(tok, 'hello')

    def test_roundtrip_difference_is_rejected(self):
        tok = TinyTokenizer()
        tok.bad_roundtrip = True
        with self.assertRaises(AssertionError):
            self.inspect(tok, 'hello')


class OfflineCliTest(unittest.TestCase):
    fixture = Path(__file__).parent / 'fixtures/qwen35_4b_config_minimal.json'

    def test_config_only_reads_directory_without_tokenizer(self):
        from week3.inspect_model import main
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_bytes(self.fixture.read_bytes())
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['--model', directory, '--config-only']), 0)
            r = json.loads(out.getvalue())
            self.assertEqual(r['kv_mib'], 128)
            self.assertTrue(r['is_teaching_fixture'])

    def test_fixture_mode_does_not_import_torch_or_transformers(self):
        code = (
            "import sys; from week3.inspect_model import main; "
            "main(['--config-only','--config-file',sys.argv[1]]); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        r = subprocess.run([sys.executable, '-B', '-c', code, str(self.fixture)],
                           text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['full_attention_layers'], 8)

    def test_cli_rejects_missing_bad_config_and_wrong_mode(self):
        from week3.inspect_model import main
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / 'config.json'
            cases = [(['--model', directory, '--config-only'], None),
                     (['--model', directory, '--config-only'], '{invalid'),
                     (['--model', directory, '--config-only'], '{}'),
                     (['--config-file', str(self.fixture)], None)]
            for args, content in cases:
                if content is not None:
                    cfg.write_text(content, encoding='utf-8')
                with self.subTest(args=args, content=content):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as exc:
                            main(args)
                    self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
