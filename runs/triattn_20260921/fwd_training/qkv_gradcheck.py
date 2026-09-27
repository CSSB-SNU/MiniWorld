"""Independent FP64 projection-attention gradient fixtures for experimental kernels."""
import torch
from torch.nn import functional as F
from native import extension
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward


def run(ext,report,save):
    base=extension('cooperative_head2')
    def relative(x,y):return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-15))
    for L in (64,128):
        for maskname in ('mixed','one_key','all_masked','dense'):
            torch.manual_seed(9261)
            z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            weights=[(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5).requires_grad_() for _ in range(3)]
            b=(torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5).requires_grad_()
            dy=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4)
            with torch.no_grad():
                if maskname=='mixed':b[...,::3]=torch.finfo(b.dtype).min
                if maskname in ('one_key','all_masked'):
                    b.fill_(torch.finfo(b.dtype).min)
                    if maskname=='one_key':b[...,7]=0
            proj=[F.linear(z,w).view(1,L,L,4,32).permute(0,3,1,2,4) for w in weights]
            bo,bm=base.forward(*proj,b)
            co,cm,cq,ck,cv=ext.forward(z,*weights,b)
            deriv=[]
            for qkv,o,m in ((proj,bo,bm),((cq,ck,cv),co,cm)):
                *dp,db=bias_backward._backward(*qkv,b,m,o,dy,True)
                dz,*dw=torch.autograd.grad(proj,(z,*weights),dp,retain_graph=True)
                deriv.append((dz,*dw,db))
            if maskname=='all_masked':
                assert all(torch.count_nonzero(x)==0 for group in deriv for x in group)
                metrics={'all_masked_zero':True}
            else:
                zr=z.detach().double().requires_grad_()
                wr=[w.detach().double().requires_grad_() for w in weights]
                br=b.detach().double().requires_grad_()
                qr,kr,vr=[F.linear(zr,w).view(1,L,L,4,32).permute(0,3,1,2,4) for w in wr]
                y=(qr@kr.transpose(-1,-2)/32**.5+br.unsqueeze(2)).softmax(-1)@vr
                ref=torch.autograd.grad(y,(zr,*wr,br),dy.double())
                metrics=[]
                for bv,cv,rv in zip(*deriv,ref):
                    assert torch.isfinite(cv).all()
                    if float(rv.norm())<1e-10:
                        be=float((bv.double()-rv).square().mean().sqrt())
                        ce=float((cv.double()-rv).square().mean().sqrt())
                        assert ce<=be*1.15+2e-7,(be,ce)
                        metrics.append(dict(zero_reference=True,baseline_rms=be,candidate_rms=ce))
                    else:
                        be=relative(bv,rv);ce=relative(cv,rv)
                        assert ce<.03 and ce<=be*1.15+.0006,(be,ce)
                        metrics.append(dict(baseline_rel=be,candidate_rel=ce))
            report['records'].append(dict(kind='fp64_fused_gradients',length=L,mask=maskname,metrics=metrics))
            save();print('FP64_QKV_GRAD',L,maskname,metrics,flush=True)
