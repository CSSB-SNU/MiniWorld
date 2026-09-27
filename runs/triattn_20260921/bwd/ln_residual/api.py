"""Isolated full-module frontend fusion; all parameters retain live autograd edges."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm import compile_native as ln
from native import extension

def _fake(dy,w,x,mean,rstd,gamma,residual,L,ending):
    return torch.empty_like(x),torch.empty_like(gamma),torch.empty_like(gamma)

@opaque(fake=_fake,name='triangle_projection_ln_residual_experiment')
def _backward(dy:list[torch.Tensor],w:list[torch.Tensor],x:torch.Tensor,mean:torch.Tensor,rstd:torch.Tensor,
              gamma:torch.Tensor,residual:torch.Tensor,L:int,ending:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    dx,dg,db=extension().backward(dy,w,x,mean,rstd,gamma,residual.contiguous(),L,ending)
    return dx.view_as(x),dg,db

class Front(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,beta,eps,ending,wq,wk,wv,wg,wb):
        xx=x.transpose(1,2).contiguous() if ending else x
        z,mean,rstd=ln._dispatch_fwd(xx,gamma,beta,eps)
        weights=(wq,wk,wv,wg,wb)
        ctx.save_for_backward(xx,z,mean,rstd,gamma,*weights)
        ctx.ending=ending
        return *(F.linear(z,w) for w in weights),x.view_as(x)
    @staticmethod
    def backward(ctx,dq,dk,dv,dg,db,residual):
        x,z,mean,rstd,gamma,*weights=ctx.saved_tensors
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip((dq,dk,dv,dg,db),weights)]
        dx,dgamma,dbeta=_backward(dy,weights,x,mean,rstd,gamma,residual,x.shape[1],ctx.ending)
        zz=z.reshape(-1,128)
        dw=[g.T@zz if needed else None for g,needed in zip(dy,ctx.needs_input_grad[5:])]
        return dx,dgamma,dbeta,None,None,*dw

def forward(model,pair,mask=None):
    B,L,_,C=pair.shape
    q,k,v,g,b,residual=Front.apply(pair,model.ln_pair.weight,model.ln_pair.bias,model.ln_pair.eps,not model.starting,
        model.to_query.weight,model.to_key.weight,model.to_value.weight,model.to_gate.weight,model.to_bias.weight)
    q,k,v=(t.view(B,L,L,4,32).permute(0,3,1,2,4) for t in (q,k,v))
    b=b.permute(0,3,1,2)
    if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
    out=model._kernel_triangle_attention(q,k,v,b,model._backend)
    out=out.permute(0,2,3,1,4).reshape(B,L,L,C)
    out=model._gate_out(g,out)
    if not model.starting:out=out.transpose(1,2).contiguous()
    if model.p_drop>0 and model.training:out=out*model._make_drop_scale(pair,model.p_drop)
    return residual+out
