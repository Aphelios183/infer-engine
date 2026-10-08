import copy
import json
import unittest

from qwen35_adapter.weight_plan import expected_text, build_plan, unique_object


class WeightPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = {'model_type': 'qwen3_5', 'text_config': {
            'model_type': 'qwen3_5_text', 'num_hidden_layers': 2,
            'hidden_size': 8, 'intermediate_size': 12, 'vocab_size': 20,
            'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4,
            'max_position_embeddings': 16, 'linear_num_key_heads': 1,
            'linear_num_value_heads': 2, 'linear_key_head_dim': 3,
            'linear_value_head_dim': 4, 'linear_conv_kernel_dim': 4,
            'layer_types': ['linear_attention', 'full_attention'],
            'tie_word_embeddings': True, 'attention_bias': False,
            'attn_output_gate': True, 'hidden_act': 'silu'}}
        self.tensors = {k: {'shape': v['shape'], 'dtype': 'BF16'}
                        for k,v in expected_text(self.config).items()}

    def test_complete_mapping_alias_and_no_mutation(self):
        before = copy.deepcopy(self.tensors)
        plan = build_plan(self.config, self.tensors)
        self.assertEqual(plan['mapped_count'], 27)
        self.assertEqual(plan['aliases'], {'lm_head.weight': 'model.embed_tokens.weight'})
        self.assertEqual(before, self.tensors)

    def test_missing_text(self):
        self.tensors.pop('model.language_model.norm.weight')
        with self.assertRaisesRegex(ValueError, 'missing='):
            build_plan(self.config, self.tensors)

    def test_extra_text_and_unknown_namespace(self):
        for key in ('model.language_model.layers.2.foo', 'unknown.weight', 'lm_head.weight'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'unexpected='):
                    build_plan(self.config, dict(self.tensors, **{key: {}}))

    def test_explicit_vision_mtp_exclusion(self):
        self.tensors.update({'model.visual.foo': {}, 'mtp.foo': {}})
        self.assertEqual(build_plan(self.config, self.tensors)['ignored_counts'], {'vision':1,'mtp':1})

    def test_wrong_q_gate_shape(self):
        self.tensors['model.language_model.layers.1.self_attn.q_proj.weight']['shape'] = [8,8]
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            build_plan(self.config, self.tensors)

    def test_layout_and_norm_semantics(self):
        specs = expected_text(self.config)
        p = 'model.language_model.layers.'
        self.assertEqual(specs[p+'1.self_attn.q_proj.weight']['semantics'], 'per_head_Q_then_gate')
        self.assertEqual(specs[p+'1.self_attn.q_norm.weight']['semantics'], 'rmsnorm_one_plus_weight')
        self.assertEqual(specs[p+'0.linear_attn.norm.weight']['semantics'], 'gated_norm_direct_weight')

    def test_unsupported_dtype(self):
        self.tensors['model.language_model.norm.weight']['dtype'] = 'I8'
        with self.assertRaisesRegex(ValueError, 'dtype'):
            build_plan(self.config, self.tensors)

    def test_unsupported_untied_model(self):
        self.config['text_config']['tie_word_embeddings'] = False
        with self.assertRaisesRegex(ValueError, 'tied'):
            build_plan(self.config, self.tensors)

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            json.loads('{"weight": 1, "weight": 2}', object_pairs_hook=unique_object)


if __name__ == '__main__':
    unittest.main()
