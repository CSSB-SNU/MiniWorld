"""Exercise actual module dispatch, all grads, optimizer updates and compile."""
import copy,json,types,os
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
import api
from native import extension,ARTIFACT_ROOT
extension()
ROWS=int(os.environ.get("FUSION_ROWS","4"))

torch.manual_seed(190222)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ROOT=Path(__file__).resolve().parent
result=[]
def save():(ROOT/('module-correctness-'+os.environ.get('SLURM_JOB_ID','local')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
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
    L=int(os.environ.get('GATE_MODULE_LENGTH','768'))
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=.25).cuda().bfloat16().train()
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    base._fuse_bias_backward=True
    cand=copy.deepcopy(base)
    cand.forward=types.MethodType(api.forward,cand)
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x)
    weights=tuple(getattr(cand,n).weight for n in ('to_query','to_key','to_value','to_gate','to_bias'))

    for maskname in ('mixed','all_masked'):
        mask=torch.ones(1,L,device='cuda',dtype=torch.bool)
        if maskname=='mixed':mask[:,::7]=False
        else:mask[:]=False
        b=evaluate(base,x,mask,dy)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:c=evaluate(cand,x,mask,dy)
        assert any('gate_delta_tma' in e.name for e in prof.events()),'native kernel did not execute'
        assert torch.equal(b[0],c[0]),'forward changed'
        errs=compare(b,c);result.append(dict(kind='module',ending=ending,mask=maskname,dropout=.25,errors=errs));save()
    # Current weights must be read after an update; no detached weight cache.
    opt0=torch.optim.SGD(base.parameters(),lr=.1);opt1=torch.optim.SGD(cand.parameters(),lr=.1)
    before=[w.detach().clone() for w in cand.parameters()]
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    for model,opt in ((base,opt0),(cand,opt1)):
        opt.zero_grad(set_to_none=True);torch.manual_seed(8888)
        ((model(x,mask)*dy).float().mean()*128).backward();opt.step()
    changed=[int((old!=new).sum()) for old,new in zip(before,cand.parameters())]
    assert all(n>0 for n in changed),changed
    errs=compare(evaluate(base,x,mask,dy),evaluate(cand,x,mask,dy))
    result.append(dict(kind='optimizer_update',ending=ending,errors=errs,changed_elements=changed));save()
    cand.p_drop=0
    ref=evaluate(cand,x,mask,dy)
    compiled=torch.compile(cand,backend='inductor',fullgraph=True)
    y=compiled(x,mask)
    val=[y.detach(),*[v.detach() for v in torch.autograd.grad(y,(x,*cand.parameters()),dy)]]
    errs=compare(ref,val)
    result.append(dict(kind='inductor_fullgraph',ending=ending,errors=errs));save()
    # Frozen projection parameters still permit a native input gradient.
    for w in cand.parameters():w.requires_grad_(False)
    dx=torch.autograd.grad(cand(x,mask),x,dy)[0]
    assert torch.isfinite(dx).all() and dx.float().norm()>0
    result.append(dict(kind='frozen_parameters',ending=ending));save()
    print('INSTALLED_PASS',ending,flush=True)

print('API_PASS',len(result),flush=True)
