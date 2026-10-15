"""小型教学对拍：FP64 CPU参考、非连续页、GQA、尾块与请求换序。无性能计时。"""
import os
import tempfile
import json
import torch


def reference(q,k,v,tables,lengths):
    # 与kernel不同实现：先按逻辑顺序gather，再一次softmax。仅作参考。
    q,k,v=q.cpu().double(),k.cpu().double(),v.cpu().double()
    tables,lengths=tables.cpu(),lengths.cpu()
    result=[];page=k.shape[1];ratio=q.shape[1]//k.shape[2]
    for row,n in enumerate(lengths.tolist()):
        t=torch.arange(n);slots=tables[row,t//page].long()*page+t%page
        keys=k.flatten(0,1)[slots];values=v.flatten(0,1)[slots]
        heads=[]
        for head in range(q.shape[1]):
            score=(keys[:,head//ratio]*q[row,head]).sum(-1)*q.shape[-1]**-0.5
            heads.append((score.softmax(0)[:,None]*values[:,head//ratio]).sum(0))
        result.append(torch.stack(heads))
    return torch.stack(result)


@torch.inference_mode()
def run():
    import triton
    from paged_decode import paged_decode
    torch.manual_seed(42);torch.set_num_threads(2)
    records=[]
    # 自己的kernel可用toy页大小4；不是FlashAttention的页大小限制。
    for dtype in (torch.float32,torch.bfloat16):
        for page,d,hq,hkv,lens in ((4,64,4,1,[1,6]),(4,80,4,2,[9,17]),
                                   (256,256,16,4,[255,257]),(256,256,16,4,[256,300])):
            width=(max(lens)+page-1)//page;pages=2*width+3
            tables=torch.full((2,width),-1,device='cuda',dtype=torch.int32)
            k=torch.full((pages,page,hkv,d),float('nan'),device='cuda',dtype=dtype)
            v=torch.full_like(k,float('nan'))
            permutation=torch.randperm(pages,device='cuda');cursor=0
            for row,n in enumerate(lens):
                needed=(n+page-1)//page
                tables[row,:needed]=permutation[cursor:cursor+needed].int();cursor+=needed
                t=torch.arange(n,device='cuda');slots=tables[row,t//page].long()*page+t%page
                k.view(-1,hkv,d)[slots]=torch.randn(n,hkv,d,device='cuda',dtype=dtype)*.3
                v.view(-1,hkv,d)[slots]=torch.randn(n,hkv,d,device='cuda',dtype=dtype)
            q=torch.randn(2,hq,d,device='cuda',dtype=dtype)
            lengths=torch.tensor(lens,device='cuda',dtype=torch.int32)
            expected=reference(q,k,v,tables,lengths).to(device='cuda',dtype=dtype)
            k_before=k.view(torch.uint8).clone();v_before=v.view(torch.uint8).clone()
            for tile in (8,16,32):
                actual=paged_decode(q,k,v,tables,lengths,block_t=tile)
                assert torch.isfinite(actual).all()
                # 预先声明：FP32严格阈值；BF16允许末次存储的舍入差异。
                atol,rtol=(2e-5,2e-5) if dtype==torch.float32 else (.008,.01)
                torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
                rev=torch.tensor([1,0],device='cuda')
                swapped=paged_decode(q[rev],k,v,tables[rev],lengths[rev],block_t=tile)
                torch.testing.assert_close(swapped,actual[rev],atol=0,rtol=0)
                assert torch.equal(k.view(torch.uint8),k_before) and torch.equal(v.view(torch.uint8),v_before)
                records.append({'dtype':str(dtype),'page':page,'D':d,'lengths':lens,'tile':tile,
                    'max_abs':(actual.float()-expected.float()).abs().max().item()})
            for bad in (torch.zeros_like(lengths),torch.full_like(lengths,width*page+1)):
                try:paged_decode(q,k,v,tables,bad)
                except ValueError:pass
                else:raise AssertionError('invalid length accepted')
            bad_table=tables.clone();bad_table[0,0]=-1
            try:paged_decode(q,k,v,bad_table,lengths)
            except ValueError:pass
            else:raise AssertionError('invalid effective page accepted')
    torch.cuda.synchronize()
    print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,
        'gpu':torch.cuda.get_device_name(0),'cases':records,'count':len(records),
        'checks':'FP64 reference, poison padding, immutable pools, reordered requests, rejected invalid metadata'},indent=2))


if __name__=='__main__':
    # 本次编译缓存仅属于此实验，退出后自动清理；不触碰用户共享缓存。
    with tempfile.TemporaryDirectory(prefix='infer-triton-lesson-') as cache:
        os.environ['TRITON_CACHE_DIR']=cache
        run()
