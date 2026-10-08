import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from nanovllm.sampling_params import SamplingParams
from qwen35_adapter.parameters import Qwen35TextParameters
from qwen35_adapter.state_manager import StateManager
from qwen35_adapter.admission import AdmissionController
from qwen35_adapter.scheduler import HybridScheduler
from qwen35_adapter.runner_inputs import RunnerInputBuilder
from qwen35_adapter.pool_backend import PooledCPUBackend
from qwen35_adapter.execution import CPUExecutionRunner
from qwen35_adapter.layer_compute import decoder_layer,rms_norm


class PoolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        patcher=patch.object(Sequence,'block_size',4)
        patcher.start(); self.addCleanup(patcher.stop)
        self.c={'model_type':'qwen3_5_text','hidden_size':8,'intermediate_size':12,
            'vocab_size':64,'num_hidden_layers':4,'num_attention_heads':2,
            'num_key_value_heads':1,'head_dim':4,'max_position_embeddings':128,
            'linear_num_key_heads':1,'linear_num_value_heads':2,'linear_key_head_dim':2,
            'linear_value_head_dim':2,'linear_conv_kernel_dim':4,
            'layer_types':['linear_attention','full_attention','linear_attention','full_attention'],
            'tie_word_embeddings':True,'attention_bias':False,'attn_output_gate':True,
            'hidden_act':'silu','rms_norm_eps':1e-6,
            'rope_parameters':{'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':0.5}}
        self.p=Qwen35TextParameters({'model_type':'qwen3_5','text_config':self.c})
        for name,param in list(self.p.named_parameters()):
            self.p.replace_parameter(name,torch.randn(param.shape)*0.1)
        self.p.tie_weights(); self.p.loaded=True # synthetic fixture, not real checkpoint
        self.bm=BlockManager(8,4)
        self.bm.free_block_ids.clear(); self.bm.free_block_ids.extend([5,2,7,0,1,3,4,6])
        self.sm=StateManager(torch.zeros(2,3,8,4),torch.zeros(2,3,2,2,2))
        self.s=HybridScheduler(AdmissionController(self.bm,self.sm,max_prefill_tokens=16),
                              max_num_seqs=2,token_budget=16,eos_ids={63})
        self.a=Sequence([11,22,33],SamplingParams(max_tokens=5))
        self.b=Sequence([51,52],SamplingParams(max_tokens=5))
        self.s.add(self.a); self.s.add(self.b)
        self.kv=torch.full((2,2,8,4,1,4),777.)
        self.backend=PooledCPUBackend(self.s,self.p,self.kv)

    def dense(self,seq):
        x=F.embedding(torch.tensor(seq.token_ids)[None],self.p.model.embed_tokens.weight)
        states=[]
        for i,node in enumerate(self.p.model.layers):
            x,st=decoder_layer(x,torch.arange(x.shape[1])[None],node,self.c,i)
            states.append(st)
        return rms_norm(x,self.p.model.norm.weight,self.c['rms_norm_eps'])[0],states

    def check_batch(self,batch):
        inp=RunnerInputBuilder(self.s).prepare(batch)
        out=self.backend.forward(inp)
        edges=inp.cu_seqlens_q.tolist()
        for row,seq in enumerate(batch.sequences):
            reference,states=self.dense(seq)
            a,b=edges[row:row+2]; n=b-a
            torch.testing.assert_close(out[a:b],reference[-n:],atol=2e-5,rtol=2e-4)
            for layer,state in enumerate(states):
                if layer in self.backend.full_map:
                    idx=self.backend.full_map[layer]
                    slots=torch.tensor([seq.block_table[j//4]*4+j%4 for j in range(seq.num_tokens)])
                    for which,value in enumerate((state.key,state.value)):
                        actual=self.kv[which,idx].view(-1,1,4)[slots]
                        torch.testing.assert_close(actual,value[0].transpose(0,1),atol=2e-5,rtol=2e-4)
                else:
                    idx=self.backend.linear_map[layer]; slot=self.sm.lookup(seq.seq_id)
                    torch.testing.assert_close(self.sm.conv_pool[idx,slot],state.conv[0],atol=2e-5,rtol=2e-4)
                    torch.testing.assert_close(self.sm.recurrent_pool[idx,slot],state.recurrent[0],atol=2e-5,rtol=2e-4)
        return inp,out

    def test_prefill_all_layer_maps_and_unused_slots(self):
        self.check_batch(self.s.schedule())
        self.assertEqual(self.backend.full_map,{1:0,3:1})
        self.assertEqual(self.backend.linear_map,{0:0,2:1})
        self.assertTrue(torch.all(self.kv[:,:,5,3]==777))
        self.assertTrue(torch.all(self.kv[:,:,2,2:]==777))
        self.assertTrue(torch.all(self.sm.conv_pool[:,2]==0))
        self.assertTrue(torch.all(self.sm.recurrent_pool[:,2]==0))

    def test_reorder_cross_block_and_old_kv_unchanged(self):
        batch=self.s.schedule(); self.check_batch(batch); self.s.postprocess(batch,[44,53])
        old=self.kv[:,:,5,:3].clone()
        self.s.running.rotate(1)
        batch=self.s.schedule(); inp,_=self.check_batch(batch)
        self.assertEqual(inp.state_slots.tolist(),[1,0])
        self.s.postprocess(batch,[54,45])
        batch=self.s.schedule(); self.check_batch(batch)
        self.assertEqual(self.a.block_table,[5,7])
        self.assertTrue(torch.equal(old,self.kv[:,:,5,:3]))
        self.assertFalse(torch.all(self.kv[:,:,7,0]==777))
        self.assertTrue(torch.all(self.kv[:,:,7,1:]==777))

    def test_changed_metadata_rejected_before_pool_write(self):
        batch=self.s.schedule(); inp=RunnerInputBuilder(self.s).prepare(batch)
        inp.state_slots[0]=2
        before=self.kv.clone()
        with self.assertRaisesRegex(ValueError,'changed'):
            self.backend.forward(inp)
        self.assertTrue(torch.equal(before,self.kv))
        self.assertTrue(torch.all(self.sm.conv_pool==0))

    def test_duplicate_forward_rejected(self):
        batch=self.s.schedule(); inp,_=self.check_batch(batch)
        old=self.sm.recurrent_pool.clone()
        with self.assertRaisesRegex(RuntimeError,'already attempted'):
            self.backend.forward(inp)
        self.assertTrue(torch.equal(old,self.sm.recurrent_pool))

    def test_runner_execution_and_eos_release(self):
        class Sampler:
            execution_mode='cpu_sync'
            def sample(self,logits,temperatures):
                assert logits.shape==(2,64)
                return torch.tensor([63,63])
        runner=CPUExecutionRunner(self.s,self.backend,Sampler())
        result=runner.run(self.s.schedule())
        self.assertEqual(result.token_ids,(63,63))
        self.assertTrue(self.s.is_finished()); self.assertEqual(self.sm.owners,{})
        self.assertEqual(len(self.bm.free_block_ids),8)

    def test_partial_layer_failure_disposes_and_reuse_clears_state(self):
        class Sampler:
            execution_mode='cpu_sync'
            def sample(self,*args): raise AssertionError('must not reach sampling')
        original=decoder_layer
        def injected(*args,**kwargs):
            if args[4]==1: raise RuntimeError('injected after GDN writes')
            return original(*args,**kwargs)
        runner=CPUExecutionRunner(self.s,self.backend,Sampler())
        with patch('qwen35_adapter.pool_backend.decoder_layer',side_effect=injected):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                runner.run(self.s.schedule())
        self.assertTrue(self.s.is_finished()); self.assertEqual(self.sm.owners,{})
        self.assertTrue(torch.any(self.sm.conv_pool!=0)) # disposal is not rollback
        self.sm.allocate('consume-unused-slot') # free slot2 first, then dirty slot0
        self.assertTrue(torch.any(self.sm.conv_pool[:,0]!=0))
        slot=self.sm.allocate('new')
        self.assertEqual(slot,0)
        self.assertTrue(torch.all(self.sm.conv_pool[:,slot]==0))
        self.assertTrue(torch.all(self.sm.recurrent_pool[:,slot]==0))

    def test_active_neighbor_not_touched_when_only_a_decodes(self):
        batch=self.s.schedule(); self.check_batch(batch); self.s.postprocess(batch,[44,53])
        conv=self.sm.conv_pool[:,1].clone(); rec=self.sm.recurrent_pool[:,1].clone()
        kv=self.kv[:,:,2].clone()
        self.s.max_num_seqs=1 # test limits this round to A, B remains active
        batch=self.s.schedule()
        self.assertEqual(batch.sequences,(self.a,))
        self.check_batch(batch)
        self.assertTrue(torch.equal(conv,self.sm.conv_pool[:,1]))
        self.assertTrue(torch.equal(rec,self.sm.recurrent_pool[:,1]))
        self.assertTrue(torch.equal(kv,self.kv[:,:,2]))

    def test_corrupt_kv_write_address_rejected(self):
        batch=self.s.schedule(); inp=RunnerInputBuilder(self.s).prepare(batch)
        inp.slot_mapping[0]=8 # points into B's block, instead of A's block5
        before=self.kv.clone()
        with self.assertRaisesRegex(ValueError,'slot_mapping'):
            self.backend.forward(inp)
        self.assertTrue(torch.equal(before,self.kv))

    def test_bad_pool_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'KV pool'):
            PooledCPUBackend(self.s,self.p,self.kv[:,:1])


if __name__=='__main__': unittest.main()
