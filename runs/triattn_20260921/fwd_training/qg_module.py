"""Experimental combined front/attention autograd; existing native backward kernels."""
import torch
from torch.nn import functional as F
from native import extension
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.layernorm import compile_native as ln
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward,gate_backward,wgrad_backward
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd as gate_fwd

_extension=None


def _fake(z,wq,wk,wv,wg,b):
    L=z.shape[1]
    def value():return torch.empty_like(z).view(1,L,L,4,32).permute(0,3,1,2,4)
    return value(),torch.empty((1,4,L,L),device=z.device,dtype=torch.float32),value(),value(),value(),torch.empty_like(z)


@opaque(fake=_fake,name='triangle_qg_attention_fused_experiment')
def native(z:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,wv:torch.Tensor,wg:torch.Tensor,b:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return tuple(_extension.forward(z,wq,wk,wv,wg,b))


def projection_view(t):
    return t.permute(0,2,3,1,4).reshape(-1,128)


def make_forward(artifact):
    global _extension
    _extension=extension(artifact)

    class FrontAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,gamma,beta,wq,wk,wv,wg,wb,wo,mask,eps,ending):
            xx=x.transpose(1,2).contiguous() if ending else x
            z,mean,rstd=ln._dispatch_fwd(xx,gamma,beta,eps)
            b=F.linear(z,wb).permute(0,3,1,2)
            if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
            b=b.contiguous()
            out,m,q,k,v,gate=native(z,wq,wk,wv,wg,b)
            gate=gate.reshape(-1,128)
            out2=projection_view(out)
            y=gate_fwd(gate,out2,wo,shape_key=token_key(x.shape[1]))
            ctx.save_for_backward(xx,z,mean,rstd,gamma,wq,wk,wv,wg,wb,wo,q,k,v,b,m,gate,out2,mask)
            ctx.ending=ending
            return y.view_as(z),x.view_as(x)

        @staticmethod
        def backward(ctx,dy,residual):
            x,z,mean,rstd,gamma,wq,wk,wv,wg,wb,wo,q,k,v,b,m,gate,out,mask=ctx.saved_tensors
            dy=dy.reshape(-1,128).contiguous()
            dr,dg,a,delta=gate_backward.gate_backward(dy,wo,gate,out)
            dq,dk,dv,db=gate_backward.attention_backward(q,k,v,b,m,delta,dr)
            dwo=dy.T@a if ctx.needs_input_grad[8] else None
            del dr,a,delta
            if mask is not None:db=db.masked_fill(~mask[:,None,None,:],0)
            weights=(wq,wk,wv,wg,wb)
            grads=[projection_view(g).contiguous() for g in (dq,dk,dv)]
            grads.extend((dg,db.permute(0,2,3,1).reshape(-1,4).contiguous()))
            dx,dgamma,dbeta=ln_backward._backward(grads,list(weights),x,mean,rstd,gamma,residual.contiguous(),x.shape[1],ctx.ending)
            zz=z.reshape(-1,128);needed=ctx.needs_input_grad[3:8]
            if all(needed[:4]) and wgrad_backward.can_use(grads[:4],zz):
                dw=[*wgrad_backward.backward(grads[:4],zz),grads[4].T@zz if needed[4] else None]
            else:dw=[g.T@zz if n else None for g,n in zip(grads,needed)]
            return dx,dgamma,dbeta,*dw,dwo,None,None,None

    def forward(model,pair,mask=None):
        out,residual=FrontAttention.apply(pair,model.ln_pair.weight,model.ln_pair.bias,
            model.to_query.weight,model.to_key.weight,model.to_value.weight,model.to_gate.weight,
            model.to_bias.weight,model.to_out.weight,mask,model.ln_pair.eps,not model.starting)
        if not model.starting:out=out.transpose(1,2).contiguous()
        if model.p_drop>0 and model.training:out=out*model._make_drop_scale(pair,model.p_drop)
        return residual+out

    return forward
