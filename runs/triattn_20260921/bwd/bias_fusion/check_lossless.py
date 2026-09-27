"""Require exact gradients against installed FP32 partials, including large exponent gaps."""
import json, os
from pathlib import Path
import torch
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward as installed
import hybrid
from native import extension

assert extension() is not installed._extension()
torch.manual_seed(92324)
results=[]
for L in (64,128):
    for mode in ('random','mixed','all_masked','wide_exponents','small','large'):
        tensors=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
        if mode=='wide_exponents':
            scale=(2.**(torch.arange(L,device='cuda')%40-20)).to(torch.bfloat16)
            tensors[-1].mul_(scale[None,:,None,None])
        if mode=='small':tensors[-1].mul_(1e-5)
        if mode=='large':tensors[-1].mul_(1e5)
        q,k,v,dy=[x.view(1,L,L,4,32).permute(0,3,1,2,4) for x in tensors]
        b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
        if mode=='mixed':b[...,::3]=torch.finfo(b.dtype).min
        if mode=='all_masked':b.fill_(torch.finfo(b.dtype).min)
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
        ref=installed._backward(q,k,v,b,m,out,dy,True)
        cand=hybrid.backward(q,k,v,b,m,out,dy,4)
        equal=[torch.equal(x,y) for x,y in zip(ref,cand)]
        assert all(torch.isfinite(x).all() for x in cand)
        result=dict(length=L,mode=mode,bitwise_equal=equal)
        results.append(result)
        path=Path(__file__).parent/('lossless-correctness-'+os.environ.get('SLURM_JOB_ID','local')+'.json')
        path.write_text(json.dumps(results,indent=2)+'\n')
        print('LOSSLESS',result,flush=True)
        assert all(equal),result
