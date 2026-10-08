import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from qwen35_adapter.parameters import Qwen35TextParameters
from qwen35_adapter.weight_loader import load_text_parameters
from qwen35_adapter.weight_plan import expected_text


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'model_type':'qwen3_5','text_config': {
            'model_type':'qwen3_5_text','num_hidden_layers':2,'hidden_size':8,
            'intermediate_size':12,'vocab_size':20,'num_attention_heads':2,
            'num_key_value_heads':1,'head_dim':4,'max_position_embeddings':16,
            'linear_num_key_heads':1,'linear_num_value_heads':2,
            'linear_key_head_dim':3,'linear_value_head_dim':4,'linear_conv_kernel_dim':4,
            'layer_types':['linear_attention','full_attention'],
            'tie_word_embeddings':True,'attention_bias':False,'attn_output_gate':True,'hidden_act':'silu'}}
        self.weights = {}
        for i,(name,spec) in enumerate(expected_text(self.config).items()):
            dtype = torch.float32 if name.endswith('A_log') else torch.bfloat16
            self.weights[name] = torch.arange(torch.tensor(spec['shape']).prod().item(), dtype=torch.float32).reshape(spec['shape']).to(dtype) + i
        self.weights['model.visual.fake'] = torch.zeros(1)
        self.weights['mtp.fake'] = torch.zeros(1)
        self.write_checkpoint()

    def write_checkpoint(self):
        (self.root/'config.json').write_text(json.dumps(self.config))
        groups = [{}, {}]
        mapping = {}
        for i,(k,v) in enumerate(self.weights.items()):
            groups[i%2][k] = v
            mapping[k] = f'part{i%2}.safetensors'
        for i,group in enumerate(groups):
            save_file(group, str(self.root/f'part{i}.safetensors'))
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))

    def test_meta_tree_and_forward_guard(self):
        model = Qwen35TextParameters(self.config)
        self.assertEqual(len(list(model.parameters())), 27)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        with self.assertRaises(NotImplementedError):
            model(torch.tensor([1]))

    def test_every_value_dtype_and_alias(self):
        model, report = load_text_parameters(self.root, verify_values=True)
        self.assertEqual(report['verified_tensors'], 27)
        self.assertTrue(model.loaded)
        for source,spec in expected_text(self.config).items():
            actual = model.get_parameter(spec['target'])
            self.assertTrue(torch.equal(actual, self.weights[source]))
            self.assertEqual(actual.dtype, self.weights[source].dtype)
            self.assertFalse(actual.requires_grad)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(len(model.state_dict()), 28)
        with self.assertRaises(NotImplementedError):
            model(torch.tensor([1]))

    def test_storage_does_not_write_checkpoint(self):
        model,_ = load_text_parameters(self.root)
        before = (self.root/'part0.safetensors').read_bytes()
        model.model.embed_tokens.weight.zero_()
        self.assertEqual(before, (self.root/'part0.safetensors').read_bytes())
        fresh,_ = load_text_parameters(self.root)
        self.assertTrue(torch.equal(fresh.model.embed_tokens.weight, self.weights['model.language_model.embed_tokens.weight']))

    def test_missing_key_rejected_before_materialization(self):
        del self.weights['model.language_model.norm.weight']
        self.write_checkpoint()
        with patch('qwen35_adapter.weight_loader.Qwen35TextParameters') as constructor:
            with self.assertRaisesRegex(ValueError,'missing='):
                load_text_parameters(self.root)
            constructor.assert_not_called()

    def test_bad_shape(self):
        self.weights['model.language_model.norm.weight'] = torch.zeros(7)
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError,'shape mismatch'):
            load_text_parameters(self.root)

    def test_failure_does_not_publish_loaded_model(self):
        original = Qwen35TextParameters.replace_parameter
        seen = []
        def fail(model,name,tensor):
            if seen:
                raise RuntimeError('injected load failure')
            seen.append(model)
            original(model,name,tensor)
        with patch.object(Qwen35TextParameters,'replace_parameter',fail):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                load_text_parameters(self.root)
        self.assertFalse(seen[0].loaded)

    def test_registered_target_mismatch(self):
        class Extra(Qwen35TextParameters):
            def __init__(self,config):
                super().__init__(config)
                self.extra = torch.nn.Parameter(torch.empty(1,device='meta'))
        with patch('qwen35_adapter.weight_loader.Qwen35TextParameters',Extra):
            with self.assertRaisesRegex(ValueError,'names'):
                load_text_parameters(self.root)


if __name__ == '__main__':
    unittest.main()
