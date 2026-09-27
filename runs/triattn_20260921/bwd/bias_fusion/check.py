"""Independent FP64 adjoints plus native determinism and full module gradients."""
import argparse,copy,json,types,os
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
import hybrid
from native import extension

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
torch.manual_seed(230923)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
extension();results=[]
row_options=[int(x) for x in os.environ.get('FUSION_ROW_OPTIONS','4 8').split()]
def save():a.output.write_text(json.dumps(results,indent=2)+'\n')
def error(x,y):return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))

def assessment(base,cand,ref):
    report=[]
    for bv,cv,rv in zip(base,cand,ref):
        assert torch.isfinite(cv).all()
        br=float((bv.double()-rv).square().mean().sqrt());cr=float((cv.double()-rv).square().mean().sqrt())
        if float(rv.norm())<1e-10:
            # Rounded forward O makes stock delta leave a tiny single-key residual.
            assert cr<=br*1.10+2e-7,(br,cr)
            report.append(dict(zero_reference=True,baseline_rms=br,candidate_rms=cr))
        else:
            be=error(bv,rv);ce=error(cv,rv)
            assert ce<.03 and ce<=be*1.15+.0006,(be,ce)
            report.append(dict(baseline_relative_l2=be,candidate_relative_l2=ce))
    return report

for L in (64,128):
    for maskname in ('none','mixed','one_key','all_masked'):
        q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
        b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
        if maskname=='mixed':b[...,::3]=torch.finfo(b.dtype).min
        if maskname in ('one_key','all_masked'):
            b.fill_(torch.finfo(b.dtype).min)
            if maskname=='one_key':b[...,7]=0
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
        bdq,bdk,bdv,bdb=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))
        base=(bdq,bdk,bdv,bdb.reshape(1,4,L,L,L).sum(2))
        ref=None
        if maskname!='all_masked':
            qr,kr,vr,br=[t.double().requires_grad_() for t in (q,k,v,b)]
            pr=torch.softmax(qr@kr.transpose(-2,-1)*(32**-.5)+br.unsqueeze(2),dim=-1)
            ref=torch.autograd.grad(pr@vr,(qr,kr,vr,br),dy.double())
        for rows in row_options:
            got=hybrid.backward(q,k,v,b,m,out,dy,rows)
            again=hybrid.backward(q,k,v,b,m,out,dy,rows)
            assert all(torch.equal(x,y) for x,y in zip(got,again)),'nondeterministic'
            if ref is None:
                assert all(torch.count_nonzero(t)==0 for t in (*base,*got))
                metrics={'all_masked_zero':True}
            else:metrics=assessment(base,got,ref)
            results.append(dict(kind='core_fp64',length=L,mask=maskname,rows=rows,metrics=metrics));save()
            print('CORE_PASS',L,maskname,rows,flush=True)
if a.native_only:raise SystemExit()

def evaluate(model,x,mask,dy):
    torch.manual_seed(781)
    y=model(x,mask)
    return [y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy)]

for ending in (False,True):
    L=64
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    base._fuse_bias_backward=False
    cand=copy.deepcopy(base);refmodel=copy.deepcopy(base).double()
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x)
    for maskname in ('none','mixed','one_key','all_masked'):
        mask=None
        if maskname!='none':
            mask=torch.ones(1,L,device='cuda',dtype=torch.bool)
            if maskname=='mixed':mask[:,::3]=False
            else:mask[:]=False
            if maskname=='one_key':mask[:,7]=True
        for pdrop in (0.,.25):
            base.p_drop=cand.p_drop=pdrop
            bv=evaluate(base,x,mask,dy)
            rv=None if pdrop or maskname=='all_masked' else evaluate(refmodel,x.detach().double().requires_grad_(),mask,dy.double())
            for rows in row_options:
                cand._kernel_triangle_attention=types.MethodType(lambda self,q,k,v,b,backend:hybrid.Attention.apply(q,k,v,b,rows),cand)
                cv=evaluate(cand,x,mask,dy)
                assert torch.equal(bv[0],cv[0]),'forward changed'
                errs=[error(c,b) if float(b.norm())>1e-10 else float((c.float()-b.float()).abs().max()) for b,c in zip(bv,cv)]
                assert max(errs)<.015,errs
                metrics=None if rv is None else assessment(bv,cv,rv)
                results.append(dict(kind='module',ending=ending,mask=maskname,dropout=pdrop,rows=rows,baseline_errors=errs,fp64=metrics));save()
                print('MODULE_PASS',ending,maskname,pdrop,rows,flush=True)
print('PASS',len(results),flush=True)
