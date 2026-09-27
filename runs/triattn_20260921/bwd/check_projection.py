"""Independent FP64 adjoints, mask/dropout parity, tails and graph capture."""
import argparse,copy,json,types
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from native import extension
import projections

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
torch.manual_seed(93222)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension();results=[]
def error(x,y):return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))
def save():a.output.write_text(json.dumps(results,indent=2)+'\n')

for rows in (1,63,64,65,127,128,129,1024):
    for scale in (.01,1.,10.):
        dy=[torch.randn(rows,c,device='cuda',dtype=torch.bfloat16)*scale for c in (128,128,128,128,4)]
        w=[torch.randn(c,128,device='cuda',dtype=torch.bfloat16)*.1 for c in (128,128,128,128,4)]
        ref=sum(g.double()@ww.double() for g,ww in zip(dy,w))
        for tile in (64,128):
            y=ext.dgrad(dy,w,tile);err=error(y,ref)
            assert torch.isfinite(y).all() and err<.0021,(rows,tile,err)
            # Identical inputs must produce identical results (no atomics).
            assert torch.equal(y,ext.dgrad(dy,w,tile))
            results.append(dict(kind='native_fp64',rows=rows,tile=tile,scale=scale,error=err))
save();print('NATIVE_PASS',len(results),flush=True)
if a.native_only:raise SystemExit()

def model(ending,length):
    z=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    z._fuse_projection_backward=False
    with torch.no_grad():
        for name,w in z.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    return z
def evaluate(z,x,mask,dy):
    torch.manual_seed(9022)
    y=z(x,mask)
    return [y.detach(),*[v.detach() for v in torch.autograd.grad(y,(x,*z.parameters()),dy)]]

for ending in (False,True):
    L=64;base=model(ending,L);cand=copy.deepcopy(base)
    cand._attention=types.MethodType(projections.attention,cand)
    refmodel=copy.deepcopy(base).double()
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x)
    for maskname in ('none','mixed','one_key','all_masked'):
        mask=None
        if maskname!='none':
            mask=torch.ones(1,L,device='cuda',dtype=torch.bool)
            if maskname=='mixed':mask[:,::3]=False
            else:mask[:]=False
            if maskname=='one_key':mask[:,7]=True
        for pdrop in (0.,.25):
            base.p_drop=cand.p_drop=pdrop
            b=evaluate(base,x,mask,dy);c=evaluate(cand,x,mask,dy)
            assert all(torch.isfinite(v).all() for v in b+c),(ending,maskname,pdrop)
            assert torch.equal(b[0],c[0]),'forward changed'
            errs=[error(v,r) for v,r in zip(c,b)]
            assert max(errs)<.015,(ending,maskname,pdrop,errs)
            record=dict(kind='module',ending=ending,mask=maskname,dropout=pdrop,baseline_error=errs)
            if not pdrop and maskname!='all_masked':
                # Float64 makes guard_dtype choose the module's pure Torch path.
                r=evaluate(refmodel,x.detach().double().requires_grad_(),mask,dy.double())
                be=[error(v,rr) for v,rr in zip(b,r)];ce=[error(v,rr) for v,rr in zip(c,r)]
                zeros=[]
                for j,(bv,cv,rv) in enumerate(zip(b,c,r)):
                    if float(rv.norm())<1e-12:
                        # A single live key makes dQ/dK/dbias analytically zero.
                        # The existing BF16 attention backward leaves tiny residuals;
                        # assess them absolutely, never divide by a zero reference.
                        br=float((bv.double()-rv).square().mean().sqrt())
                        cr=float((cv.double()-rv).square().mean().sqrt())
                        change=float((cv.double()-bv.double()).square().mean().sqrt())
                        assert cr<=br*1.01+1e-8 and change<=br*.005+1e-8,(j,br,cr,change)
                        zeros.append(dict(index=j,baseline_rms=br,candidate_rms=cr,change_rms=change,
                            baseline_max=float(bv.abs().max()),candidate_max=float(cv.abs().max()),
                            criterion='no regression in existing attention residual; not a relative error against zero'))
                        be[j]=ce[j]=None
                    else:
                        assert be[j]<.05 and ce[j]<.05,(j,be,ce)
                        assert ce[j]<=be[j]*1.10+.0005,(j,be,ce)
                record.update(fp64_baseline=be,fp64_candidate=ce,zero_reference=zeros)
            results.append(record);save();print('MODULE_PASS',ending,maskname,pdrop,max(errs),flush=True)
print('PASS',len(results),flush=True)
