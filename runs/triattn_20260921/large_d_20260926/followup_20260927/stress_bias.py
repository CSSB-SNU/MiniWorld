"""Check compact partials retain BF16 range at large dO magnitudes."""
import json
import os
from pathlib import Path
import sys
import torch
from build import extension
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
import candidate
records=[];L=64
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
for D in (64,128):
    candidate.preload(D);ext=extension('bias16',D*4)
    for magnitude in (1,65536):
        torch.manual_seed(92799)
        q,k,v=[torch.randn(1,L,L,4*D,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,D).permute(0,3,1,2,4) for _ in range(3)]
        v.mul_(4)
        b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.3;b[...,::7]=torch.finfo(b.dtype).min
        dy=torch.randn(1,L,L,4*D,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,D).permute(0,3,1,2,4)*magnitude
        dy=candidate.projection_layout(dy)
        o,m=candidate.forward(q,k,v,b);delta=candidate.aux_extension().delta(o,dy)
        old=candidate.extension(D,'dkdv').backward(q,k,v,b,m,delta,dy,ext.row_group)
        new=ext.backward(q,k,v,b,m,delta,dy,ext.row_group)
        assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1])
        qr,kr,vr,br=[x.double().requires_grad_() for x in (q,k,v,b)]
        score=torch.einsum('bhijd,bhikd->bhijk',qr,kr)*(D**-.5)+br[:,:,None,:,:]
        prob=score.softmax(-1);ref=torch.einsum('bhijk,bhikd->bhijd',prob,vr)
        target=torch.autograd.grad(ref,br,dy.double())[0]
        assert torch.isfinite(new[2]).all()
        err=float((new[2].double()-target).norm()/target.norm())
        prior=float((old[2].double()-target).norm()/target.norm())
        incremental=float((new[2].float()-old[2].float()).norm()/old[2].float().norm())
        assert err<.02 and incremental<.005,(err,incremental)
        row=dict(head_dim=D,dy_magnitude=magnitude,old_fp64_relative_l2=prior,new_fp64_relative_l2=err,incremental_relative_l2=incremental,dkdv_bitwise=True,max_db=float(new[2].abs().max()))
        print('STRESS_PASS',row,flush=True);records.append(row)
(ROOT/f'bias-stress-{os.getenv("SLURM_JOB_ID")}.json').write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')
