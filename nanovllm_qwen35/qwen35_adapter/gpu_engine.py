"""Single-device eager Qwen3.5 text engine using nano request/resource classes.

Explicit adapter entry, not a replacement for upstream Qwen3 LLMEngine.
No TP/Graph/prefix/preemption/chunk scheduling. Serial owner-thread only.
"""
import math
import torch
from transformers import AutoTokenizer

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from nanovllm.sampling_params import SamplingParams
from .contracts import parse_text_contract
from .state_manager import StateManager
from .admission import AdmissionController
from .scheduler import HybridScheduler
from .runner_inputs import RunnerInputBuilder
from .pool_backend import PooledCPUBackend
from .execution import StepResult
from .weight_loader import load_text_parameters


class GPUStateManager(StateManager):
    device_type='cuda'

    def __init__(self,*args):
        super().__init__(*args)
        self.poisoned=False

    def synchronize(self):
        if self.poisoned: raise RuntimeError('GPU state manager poisoned; restart engine')
        try: torch.cuda.synchronize(self.conv_pool.device)
        except BaseException:
            self.poisoned=True
            raise

    def _initialize_slot(self,slot):
        self.synchronize()
        try:
            super()._initialize_slot(slot)
            self.synchronize() # initialization complete before ownership commit
        except BaseException:
            self.poisoned=True
            raise

    def release(self,request_id):
        self.synchronize() # no reuse while any work on the device may access pools
        super().release(request_id)


class PooledGPUBackend(PooledCPUBackend):
    execution_mode='cuda_eager'
    device_type='cuda'

    @torch.inference_mode()
    def compute_logits(self,last_hidden):
        # Correctness-first reference route: preserve per-request GEMM shape.
        # Batched BF16 LM-head GEMM can choose a different numeric algorithm.
        return torch.cat([super(PooledGPUBackend,self).compute_logits(row[None])
                          for row in last_hidden],dim=0)


class GPUExecutionRunner:
    def __init__(self,scheduler,backend,*,greedy=True):
        if backend.execution_mode!='cuda_eager': raise ValueError('GPU backend required')
        self.scheduler=scheduler; self.backend=backend; self.greedy=greedy
        self.builder=RunnerInputBuilder(scheduler)
        self.poisoned=False
        self.last_logits=None # opt-in diagnostic capture only
        self.capture_logits=False

    @torch.inference_mode()
    def run(self,batch):
        if self.poisoned or self.scheduler.admission.states.poisoned:
            raise RuntimeError('execution is poisoned; cannot run or recycle')
        inputs=self.builder.prepare(batch)
        try:
            hidden=self.backend.forward(inputs)
            logits=self.backend.compute_logits(hidden.index_select(0,inputs.last_indices.to(hidden.device)))
            if logits.ndim!=2 or logits.shape[0]!=len(batch.sequences) or not torch.isfinite(logits).all():
                raise ValueError('invalid logits')
            if self.greedy:
                ids=logits.argmax(-1)
            else:
                temps=torch.tensor([s.temperature for s in batch.sequences],device=logits.device)
                probs=(logits.float()/temps[:,None]).softmax(-1)
                ids=torch.multinomial(probs,1).squeeze(-1)
            if self.capture_logits: self.last_logits=logits.detach().cpu()
            tokens=tuple(ids.tolist())
            self.scheduler.admission.states.synchronize()
        except BaseException as exc:
            try: self.scheduler.admission.states.synchronize()
            except BaseException:
                self.poisoned=True # retain pending batch, never recycle uncertain storage
                raise RuntimeError('GPU quiescence unknown; resources quarantined') from exc
            self.scheduler.fail_batch(batch,str(exc),execution_complete=True)
            raise
        self.scheduler.postprocess(batch,tokens)
        return StepResult(inputs.request_ids,tokens,batch.input_counts,batch.is_prefill)


class Qwen35Engine:
    def __init__(self,model_dir,*,device='cuda:0',max_model_len=512,max_num_seqs=2,
                 block_size=256,greedy=True):
        if torch.device(device).type!='cuda': raise ValueError('single CUDA device required')
        if any(type(v) is not int or v<=0 for v in (max_model_len,max_num_seqs,block_size)):
            raise ValueError('capacities must be positive integers')
        self.device=torch.device(device)
        torch.cuda.set_device(self.device)
        self.max_model_len=max_model_len; self.block_size=block_size
        self.closed=False
        self.parameters,_=load_text_parameters(model_dir)
        self.parameters.to(self.device);self.parameters.tie_weights()
        self.parameters.config['text_config']['gdn_prefill_backend']='hf_torch_chunk'
        c=self.parameters.config['text_config']; contract=parse_text_contract(self.parameters.config)
        dtype=self.parameters.model.embed_tokens.weight.dtype
        capacity=max_num_seqs
        self.states=GPUStateManager(
            torch.zeros(len(contract['linear_layer_to_state']),capacity,*contract['conv_shape_per_layer_per_request'],device=self.device,dtype=dtype),
            torch.zeros(len(contract['linear_layer_to_state']),capacity,*contract['recurrent_shape_per_layer_per_request'],device=self.device,dtype=torch.bfloat16))
        blocks=max_num_seqs*math.ceil(max_model_len/block_size)
        self.kv_pool=torch.empty(2,len(contract['full_layer_to_kv']),blocks,block_size,
                                 c['num_key_value_heads'],c['head_dim'],device=self.device,dtype=dtype)
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        eos=c['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
        if self.tokenizer.eos_token_id is not None:eos.add(self.tokenizer.eos_token_id)
        self.scheduler=HybridScheduler(AdmissionController(BlockManager(blocks,block_size),self.states,
                                       max_prefill_tokens=max_model_len*max_num_seqs),
                                       max_num_seqs=max_num_seqs,token_budget=max_model_len*max_num_seqs,eos_ids=eos)
        self.backend=PooledGPUBackend(self.scheduler,self.parameters,self.kv_pool)
        self.runner=GPUExecutionRunner(self.scheduler,self.backend,greedy=greedy)
        self.states.synchronize()

    def _ready(self):
        if self.closed or self.runner.poisoned or self.states.poisoned:
            raise RuntimeError('engine unavailable; restart required')

    def add_request(self,prompt,sampling_params=None):
        self._ready()
        sp=sampling_params or SamplingParams()
        ids=self.tokenizer(prompt,add_special_tokens=False)['input_ids'] if isinstance(prompt,str) else list(prompt)
        if (not ids or any(type(x)is not int or not 0<=x<self.parameters.config['text_config']['vocab_size'] for x in ids)
                or type(sp.max_tokens)is not int or sp.max_tokens<0
                or len(ids)+sp.max_tokens>self.max_model_len):
            raise ValueError('invalid IDs or prompt+generation exceeds max_model_len')
        seq=Sequence(ids,sp);seq.block_size=self.block_size # no global class mutation
        self.scheduler.add(seq)
        return seq

    def step(self):
        self._ready()
        batch=self.scheduler.schedule()
        self._ready() # scheduling may encounter a GPU initialization failure
        return None if batch is None else self.runner.run(batch)

    def generate(self,prompts,sampling_params=None):
        self._ready()
        if not self.scheduler.is_finished():raise RuntimeError('generate requires idle engine')
        requests=[]
        try:
            for prompt in prompts:requests.append(self.add_request(prompt,sampling_params))
            while not self.scheduler.is_finished():self.step()
        except BaseException:
            if not self.runner.poisoned and not self.states.poisoned and self.scheduler._pending is None:
                for s in requests:
                    if not s.is_finished:self.scheduler.cancel(s.seq_id)
            raise
        output=[]
        for s in requests:
            result=self.scheduler.completed[s.seq_id]
            if result.reason=='failed':raise RuntimeError(result.error)
            output.append({'token_ids':list(s.completion_token_ids),
                           'text':self.tokenizer.decode(s.completion_token_ids,skip_special_tokens=True),
                           'stop_reason':result.reason})
        return output

    def close(self):
        self._ready();self.states.synchronize()
        if self.scheduler._pending is not None:raise RuntimeError('pending batch must be settled first')
        for s in tuple(self.scheduler.waiting)+tuple(self.scheduler.running):self.scheduler.cancel(s.seq_id)
        self.closed=True
        self.runner=None;self.backend=None;self.scheduler=None;self.states=None
        self.parameters=None;self.kv_pool=None
