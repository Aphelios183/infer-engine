"""Serialized CPU eager backend: real layer math with paged KV/state-pool IO.

Gathering history creates temporary dense tensors. This is a correctness bridge,
not a paged-attention kernel or production GPU implementation.
"""
import torch
import torch.nn.functional as F

from .contracts import parse_text_contract
from .runner_inputs import RunnerInputBuilder
from .layer_compute import FullState, GDNState, PagedDecode, decoder_layer, rms_norm


class PooledCPUBackend:
    execution_mode = 'cpu_sync'
    device_type = 'cpu'

    def __init__(self, scheduler, parameters, kv_pool):
        if not parameters.loaded:
            raise ValueError('parameters must be completely loaded')
        self.scheduler = scheduler
        self.parameters = parameters
        self.config = parameters.config['text_config']
        contract = parse_text_contract(parameters.config)
        self.full_map = contract['full_layer_to_kv']
        self.linear_map = contract['linear_layer_to_state']
        self.states = scheduler.admission.states
        bm = scheduler.admission.blocks
        self.block_size = bm.block_size
        c = self.config
        expected = (2,len(self.full_map),len(bm.blocks),bm.block_size,
                    c['num_key_value_heads'],c['head_dim'])
        dtype = parameters.model.embed_tokens.weight.dtype
        self.device=parameters.model.embed_tokens.weight.device
        if (kv_pool.shape != expected or kv_pool.device.type != self.device_type or kv_pool.device != self.device
                or kv_pool.dtype != dtype or kv_pool.requires_grad or not kv_pool.is_contiguous()):
            raise ValueError('invalid KV pool shape/device/dtype/layout')
        conv_shape = (len(self.linear_map),self.states.capacity,*contract['conv_shape_per_layer_per_request'])
        rec_shape = (len(self.linear_map),self.states.capacity,*contract['recurrent_shape_per_layer_per_request'])
        if (self.states.conv_pool.shape != conv_shape or self.states.recurrent_pool.shape != rec_shape
                or self.states.conv_pool.dtype != dtype
                or self.states.recurrent_pool.dtype not in (torch.float32,torch.bfloat16)):
            raise ValueError('invalid linear pool shape/dtype')
        if self.states.conv_pool.device != self.device or self.states.recurrent_pool.device != self.device:
            raise ValueError('state pools must share model device')
        for p in parameters.parameters():
            if p.device != self.device or p.requires_grad:
                raise ValueError('CPU inference parameters required')
        stores = [kv_pool,self.states.conv_pool,self.states.recurrent_pool]
        ptrs = [x.untyped_storage().data_ptr() for x in stores]
        if len(set(ptrs)) != 3 or any(p.untyped_storage().data_ptr() in ptrs for p in parameters.parameters()):
            raise ValueError('pools must not alias each other or parameters')
        self.kv_pool = kv_pool
        self._attempted_batch = None

    def _check_inputs(self, inputs):
        batch = self.scheduler._pending
        fresh = RunnerInputBuilder(self.scheduler).prepare(batch)
        for name, expected in vars(fresh).items():
            actual = getattr(inputs,name,None)
            if isinstance(expected,torch.Tensor):
                if (not isinstance(actual,torch.Tensor) or actual.device != expected.device
                        or actual.dtype != expected.dtype or not torch.equal(actual,expected)):
                    raise ValueError(f'stale or changed runner input: {name}')
            elif actual != expected:
                raise ValueError(f'stale or changed runner input: {name}')
        if batch is self._attempted_batch:
            raise RuntimeError('batch already attempted; never repeat forward on updated state')
        return batch

    @torch.inference_mode()
    def forward(self, inputs):
        batch = self._check_inputs(inputs)
        use_paged = getattr(self,'decode_backend','dense') == 'flash_paged' and not inputs.is_prefill
        if use_paged and (self.device.type != 'cuda' or self.block_size % 256
                          or any(n != 1 for n in batch.input_counts)):
            raise ValueError('flash_paged requires CUDA single-token decode and page size divisible by 256')
        # From this point failure requires whole-batch disposal, not a retry.
        self._attempted_batch = batch
        # Batch-local immutable metadata: upload once, share across all layers.
        # Do not retain/reuse these buffers across batches or change stream semantics.
        positions_gpu = inputs.positions.to(self.device)
        slots_gpu = inputs.slot_mapping.long().to(self.device)
        tables_gpu = inputs.block_tables.to(self.device) if use_paged else None
        lengths_gpu = inputs.context_lens.to(self.device) if use_paged else None
        x = F.embedding(inputs.input_ids.to(self.device),self.parameters.model.embed_tokens.weight)
        edges = inputs.cu_seqlens_q.tolist()
        # Request-local logical-to-physical history addresses do not depend on layer.
        # Rebuild every forward: row order, prefix length and block ownership may change.
        history_addresses = []
        for row, (a, b) in enumerate(zip(edges, edges[1:])):
            past = inputs.context_lens[row].item() - (b - a)
            if past and self.full_map and not use_paged:
                logical = torch.arange(past, dtype=torch.int64)
                addresses = (inputs.block_tables[row, logical // self.block_size].long()
                             * self.block_size + logical % self.block_size)
                history_addresses.append(addresses.to(self.device))
            else:
                history_addresses.append(None)  # no historical KV to gather
        c = self.config
        for layer_id, params in enumerate(self.parameters.model.layers):
            pieces = []
            for row, request_id in enumerate(inputs.request_ids):
                a,b = edges[row:row+2]
                end = inputs.context_lens[row].item()
                past = end-(b-a)
                positions = positions_gpu[a:b].unsqueeze(0)
                state_slot = inputs.state_slots[row].item()
                if self.states.lookup(request_id) != state_slot:
                    raise ValueError('state owner changed during execution')
                paged = None
                if layer_id in self.full_map:
                    idx = self.full_map[layer_id]
                    # Flatten physical blocks; logical order comes ONLY from block_table.
                    kpool = self.kv_pool[0,idx].view(-1,c['num_key_value_heads'],c['head_dim'])
                    vpool = self.kv_pool[1,idx].view_as(kpool)
                    addresses = history_addresses[row]
                    old = None if past==0 or use_paged else FullState(
                        kpool.index_select(0,addresses).transpose(0,1).unsqueeze(0),
                        vpool.index_select(0,addresses).transpose(0,1).unsqueeze(0))
                    if use_paged:
                        paged = PagedDecode(self.kv_pool[0,idx],self.kv_pool[1,idx],
                                            slots_gpu[a:b],tables_gpu[row:row+1],
                                            lengths_gpu[row:row+1],past)
                else:
                    idx = self.linear_map[layer_id]
                    old = None if past==0 else GDNState(
                        self.states.conv_pool[idx,state_slot].unsqueeze(0),
                        self.states.recurrent_pool[idx,state_slot].unsqueeze(0),past)
                out,new = decoder_layer(x[a:b].unsqueeze(0),positions,params,c,layer_id,old,
                                        state_dtype=self.states.recurrent_pool.dtype,paged=paged)
                if layer_id in self.full_map and not use_paged:
                    slots = slots_gpu[a:b]
                    # Only append current tokens. Do not rewrite historical or padding slots.
                    kpool.index_copy_(0,slots,new.key[0,:,past:,:].transpose(0,1))
                    vpool.index_copy_(0,slots,new.value[0,:,past:,:].transpose(0,1))
                elif layer_id in self.linear_map:
                    # Integer indexing gives pool views. Explicit copy_ writes originals.
                    self.states.conv_pool[idx,state_slot].copy_(new.conv[0])
                    self.states.recurrent_pool[idx,state_slot].copy_(new.recurrent[0])
                pieces.append(out[0])
            x = torch.cat(pieces,dim=0)
        return rms_norm(x,self.parameters.model.norm.weight,c['rms_norm_eps'])

    @torch.inference_mode()
    def compute_logits(self, last_hidden):
        return F.linear(last_hidden,self.parameters.lm_head.weight)
