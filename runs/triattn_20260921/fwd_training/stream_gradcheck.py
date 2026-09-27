"""Independent FP64 attention and gate-projection Jacobian fixtures."""
import torch
from torch.nn import functional as F
from native import extension
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward

def run(ext,report,save):
    base=extension('qg_scoped')
    def rel(x,y):return float((x.detach().double()-y.detach().double()).norm()/y.detach().double().norm().clamp_min(1e-15))
    for L in (64,128):
        view=lambda x:x.view(1,L,L,4,32).permute(0,3,1,2,4)
        for maskname in ('mixed','one_key','all_masked','dense'):
            torch.manual_seed(92620)
            z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            weights=[(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5).requires_grad_() for _ in range(4)]
            b=(torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5).requires_grad_()
            dy,dg=[view(torch.randn_like(z)) for _ in range(2)]
            with torch.no_grad():
                if maskname=='mixed':b[...,::3]=torch.finfo(b.dtype).min
                if maskname in ('one_key','all_masked'):
                    b.fill_(torch.finfo(b.dtype).min)
                    if maskname=='one_key':b[...,7]=0
            proj=[view(F.linear(z,w)) for w in weights]
            bo,bm,bq,bk,bv,bg=base.forward(z,*weights,b)
            bp=(bq,bk,bv)
            co,cm,cq,ck,cv,cg=ext.forward(z,*weights,b)
            assert all(torch.equal(x,y) for x,y in zip((cq,ck,cv,view(cg)),proj))
            deriv=[]
            for qkv,o,m in ((bp,bo,bm),((cq,ck,cv),co,cm)):
                *dp,db=bias_backward._backward(*qkv,b,m,o,dy,True)
                dz,*dw=torch.autograd.grad(proj,(z,*weights),(*dp,dg),retain_graph=True)
                deriv.append((dz,*dw,db))
            zr=z.detach().double().requires_grad_()
            wr=[w.detach().double().requires_grad_() for w in weights]
            br=b.detach().double().requires_grad_()
            qr,kr,vr,gr=[view(F.linear(zr,w)) for w in wr]
            if maskname=='all_masked':
                y=(qr+kr+vr)*0+br.sum()*0
                assert all(torch.count_nonzero(t)==0 for arm in deriv for t in (*arm[1:4],arm[-1]))
            else:y=(qr@kr.transpose(-1,-2)/32**.5+br.unsqueeze(2)).softmax(-1)@vr
            ref=torch.autograd.grad((y,gr),(zr,*wr,br),(dy.double(),dg.double()))
            metrics=[]
            for bv,cv,rv in zip(*deriv,ref):
                assert torch.isfinite(cv).all()
                if float(rv.norm())<1e-10:
                    be=float((bv.double()-rv).square().mean().sqrt());ce=float((cv.double()-rv).square().mean().sqrt())
                    assert ce<=be*1.15+2e-7,(be,ce)
                    metrics.append(dict(zero_reference=True,baseline_rms=be,candidate_rms=ce))
                else:
                    be=rel(bv,rv);ce=rel(cv,rv)
                    assert ce<.03 and ce<=be*1.15+.0006,(be,ce)
                    metrics.append(dict(baseline_rel=be,candidate_rel=ce))
            report['records'].append(dict(kind='fp64_stream_gradients',length=L,mask=maskname,metrics=metrics));save()
            print('FP64_QG_GRAD',L,maskname,metrics,flush=True)
