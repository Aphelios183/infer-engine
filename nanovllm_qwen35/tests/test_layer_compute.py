"""Independent CPU formulas versus installed Transformers 5.8 reference layers.

Random tiny weights; FP32 layer tolerance atol=2e-5, rtol=2e-4, NOT 4B approval.
"""
import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as ref
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache

from qwen35_adapter.layer_compute import (
    rms_norm,gated_rms_norm,split_q_gate,partial_rope,full_attention,gated_delta_net,decoder_layer)
from qwen35_adapter.parameters import Qwen35TextParameters


class LayerComputeTests(unittest.TestCase):
    observed_max_errors = []

    @classmethod
    def tearDownClass(cls):
        print('FP32 layer comparison max_abs_error =', max(cls.observed_max_errors, default=0.0))

    def setUp(self):
        torch.manual_seed(42)
        self.c = Qwen3_5TextConfig(
            hidden_size=16, intermediate_size=24, num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            linear_num_key_heads=1,linear_num_value_heads=2,
            linear_key_head_dim=4,linear_value_head_dim=4,linear_conv_kernel_dim=4,
            layer_types=['linear_attention','full_attention'],vocab_size=32,
            rope_parameters={'rope_type':'default','rope_theta':10000.,
                             'partial_rotary_factor':0.5,'mrope_section':[1,1,0]},
            attention_bias=False,attn_output_gate=True,tie_word_embeddings=True)
        self.c._attn_implementation = 'eager'
        self.t = self.c.to_dict()
        self.x = torch.randn(1,7,16)*0.3

    def close(self,a,b):
        self.assertTrue(torch.isfinite(a).all())
        self.observed_max_errors.append((a-b).abs().max().item())
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)

    def test_norm_matches_reference_fp32_and_bf16(self):
        for dtype in (torch.float32,torch.bfloat16):
            x = self.x.to(dtype)
            norm = ref.Qwen3_5RMSNorm(16,eps=1e-6)
            with torch.no_grad(): norm.weight.copy_(torch.randn(16)*0.1)
            torch.testing.assert_close(rms_norm(x,norm.weight,1e-6),norm(x),rtol=0,atol=0)
            gn = ref.Qwen3_5RMSNormGated(16,eps=1e-6)
            gate = torch.randn_like(x)
            torch.testing.assert_close(gated_rms_norm(x,gn.weight,gate,1e-6),gn(x,gate),rtol=0,atol=0)

    def test_q_gate_interleaving(self):
        q,g = split_q_gate(torch.arange(8),2,2)
        self.assertEqual(q.flatten().tolist(),[0,1,4,5])
        self.assertEqual(g.flatten().tolist(),[2,3,6,7])
        with self.assertRaises(ValueError): split_q_gate(torch.arange(7),2,2)

    def test_partial_rope_reference_and_unchanged_tail(self):
        for dtype in (torch.float32,torch.bfloat16):
            q,k = torch.randn(1,2,3,8).to(dtype),torch.randn(1,1,3,8).to(dtype)
            pos = torch.tensor([[0,7,1024]])
            cos,sin = ref.Qwen3_5TextRotaryEmbedding(self.c)(q,pos)
            expected = ref.apply_rotary_pos_emb(q,k,cos,sin)
            actual = partial_rope(q,k,pos,theta=10000.,rotary_dim=4)
            for a,b,original in zip(actual,expected,(q,k)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertTrue(torch.equal(a[...,4:],original[...,4:]))

    def test_invalid_rope(self):
        q = torch.zeros(1,2,1,8)
        for dim in (0,3,10):
            with self.assertRaises(ValueError):
                partial_rope(q,q,torch.tensor([[0]]),theta=10000.,rotary_dim=dim)

    def test_full_reference_prefill_and_decode(self):
        module = ref.Qwen3_5Attention(self.c,1).eval()
        cache = DynamicCache(config=self.c)
        state = None
        for start,end in [(0,5),(5,6),(6,7)]:
            x,pos = self.x[:,start:end],torch.arange(start,end).unsqueeze(0)
            allowed = torch.arange(end)[None,:] <= pos[0,:,None]
            mask = torch.zeros(1,1,end-start,end).masked_fill(~allowed,torch.finfo(x.dtype).min)
            rope = ref.Qwen3_5TextRotaryEmbedding(self.c)(x,pos)
            expected,_ = module(x,rope,mask,past_key_values=cache)
            actual,state = full_attention(x,pos,module,self.t,state)
            self.close(actual,expected)
            self.close(state.key,cache.layers[1].keys)
            self.close(state.value,cache.layers[1].values)

    def test_full_cached_matches_recompute_and_preserves_old(self):
        module = ref.Qwen3_5Attention(self.c,1).eval()
        all_out,_ = full_attention(self.x,torch.arange(7)[None],module,self.t)
        _,state = full_attention(self.x[:,:5],torch.arange(5)[None],module,self.t)
        before = state.key.clone()
        out,_ = full_attention(self.x[:,5:],torch.arange(5,7)[None],module,self.t,state)
        self.close(out,all_out[:,5:])
        self.assertTrue(torch.equal(state.key,before))
        with self.assertRaises(ValueError):
            full_attention(self.x[:,5:],torch.arange(2)[None],module,self.t,state)

    def test_gdn_reference_prefill_decode_and_states(self):
        module = ref.Qwen3_5GatedDeltaNet(self.c,0).eval()
        cache = DynamicCache(config=self.c)
        state = None
        for start,end in [(0,5),(5,6),(6,7)]:
            x = self.x[:,start:end]
            expected = module(x,cache_params=cache)
            actual,state = gated_delta_net(x,module,self.t,state)
            self.close(actual,expected)
            self.close(state.conv,cache.layers[0].conv_states)
            self.close(state.recurrent,cache.layers[0].recurrent_states)
            self.assertEqual(state.length,end)

    def test_gdn_chunk_boundary_65_tokens_against_reference(self):
        module = ref.Qwen3_5GatedDeltaNet(self.c,0).eval()
        x = torch.randn(1,65,16)*0.2
        cache = DynamicCache(config=self.c)
        expected = module(x,cache_params=cache)
        actual,state = gated_delta_net(x,module,self.t)
        self.close(actual,expected)
        self.close(state.recurrent,cache.layers[0].recurrent_states)

    def test_gdn_continuation_and_old_state_unchanged(self):
        module = ref.Qwen3_5GatedDeltaNet(self.c,0).eval()
        full,final = gated_delta_net(self.x,module,self.t)
        _,state = gated_delta_net(self.x[:,:2],module,self.t)
        conv,rec = state.conv.clone(),state.recurrent.clone()
        part,tail = gated_delta_net(self.x[:,2:],module,self.t,state)
        self.close(part,full[:,2:]); self.close(tail.recurrent,final.recurrent)
        self.assertTrue(torch.equal(conv,state.conv)); self.assertTrue(torch.equal(rec,state.recurrent))

    def test_gdn_short_prompt_and_explicit_state_dtype(self):
        module = ref.Qwen3_5GatedDeltaNet(self.c,0).eval()
        for length in (1,2,3):
            expected = module(self.x[:,:length],cache_params=DynamicCache(config=self.c))
            actual,state = gated_delta_net(self.x[:,:length],module,self.t,state_dtype=torch.bfloat16)
            self.close(actual,expected)
            self.assertEqual(state.recurrent.dtype,torch.bfloat16)
            with self.assertRaises(ValueError):
                gated_delta_net(self.x[:,:1],module,self.t,state) # policy mismatch

    def test_decoder_full_and_gdn_from_parameter_skeleton(self):
        wrapper = {'model_type':'qwen3_5','text_config':self.t}
        params = Qwen35TextParameters(wrapper)
        for idx in (0,1):
            module = ref.Qwen3_5DecoderLayer(self.c,idx).eval()
            for name,value in module.named_parameters():
                params.replace_parameter(f'model.layers.{idx}.{name}',value.detach().clone())
            pos = torch.arange(7)[None]
            mask = torch.zeros(1,1,7,7).masked_fill(~torch.ones(7,7,dtype=torch.bool).tril(),torch.finfo(torch.float32).min)
            rope = ref.Qwen3_5TextRotaryEmbedding(self.c)(self.x,pos)
            expected = module(self.x,rope,attention_mask=mask if idx==1 else None,
                              past_key_values=DynamicCache(config=self.c))
            actual,_ = decoder_layer(self.x,pos,params.model.layers[idx],self.t,idx)
            self.close(actual,expected)

    def test_multi_request_rejected_not_silently_mixed(self):
        module = ref.Qwen3_5GatedDeltaNet(self.c,0).eval()
        with self.assertRaises(ValueError): gated_delta_net(self.x.expand(2,-1,-1),module,self.t)


if __name__ == '__main__':
    unittest.main()
