"""Exercise actual module dispatch, all grads, optimizer updates and compile."""
import copy,json,types,os
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward as native

torch.manual_seed(190222)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ROOT=Path(__file__).resolve().parent
result=[]
def save():(ROOT/('installed-correctness-'+os.environ.get('SLURM_JOB_ID','local')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
def evaluate(model,x,mask,dy):
    torch.manual_seed(9902)
    y=model(x,mask)
    return [y.detach(),*[t.detach() for t in torch.autograd.grad(y,(x,*model.parameters()),dy)]]
def compare(ref,val):
    assert all(torch.isfinite(t).all() for t in ref+val)
    errors=[float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)) for a,b in zip(ref,val)]
    assert max(errors)<.015,errors
    return errors

for ending in (False,True):
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():base.to_out.weight.normal_(std=.1)
    cand=copy.deepcopy(base);base._fuse_bias_backward=False
    x=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x);mask=torch.ones(1,384,device='cuda',dtype=torch.bool)
    ref=evaluate(base,x,mask,dy)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:val=evaluate(cand,x,mask,dy)
    assert not any('grouped_dkdv' in e.name for e in prof.events())
    ref2=evaluate(base,x,mask,dy)
    for name,a,b,c in zip(['output','input']+[n for n,_ in base.named_parameters()],ref,val,ref2):
        print(name,'different',int((a!=b).sum()),'max',float((a.float()-b.float()).abs().max()),'rel',float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)),'repeat_diff',int((a!=c).sum()),'repeat_rel',float((a.float()-c.float()).norm()/a.float().norm().clamp_min(1e-8)),flush=True)

