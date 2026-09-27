"""Shared attention/gate autograd boundary that passes native delta directly."""
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward, dq_backward, ln_backward
from native import extension

def _gate_fake(dy,w,gate,out):
    return (torch.empty_like(gate),torch.empty_like(gate),torch.empty_like(gate),
            torch.empty((4,gate.shape[0]),device=gate.device,dtype=torch.float32))

@opaque(fake=_gate_fake,name='triangle_gate_delta_experiment')
def gate_backward(dy:torch.Tensor,w:torch.Tensor,gate:torch.Tensor,out:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return tuple(extension().backward(dy,w,gate,out))

def _attention_fake(q,k,v,b,m,delta,dy):
    B,H,L,_,D=q.shape
    def grad():return torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
    return grad(),grad(),grad(),torch.empty_like(b)

@opaque(fake=_attention_fake,name='triangle_attention_precomputed_delta_experiment')
def attention_backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,b:torch.Tensor,m:torch.Tensor,delta:torch.Tensor,dy:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    L=q.shape[2]
    delta=delta.view(1,4,L,L)
    dy=dy.view(1,L,L,4,32).permute(0,3,1,2,4)
    dk,dv,db=bias_backward._extension().backward(q,k,v,b,m,delta,dy,4)
    dq=dq_backward.backward(q,k,v,b,m,delta,dy)
    return dq,dk,dv,db

class AttentionGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,gate,w):
        L=q.shape[2];b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
        out2=out.permute(0,2,3,1,4).reshape(-1,128).contiguous()
        gate2=gate.reshape(-1,128).contiguous()
        y=_fwd(gate2,out2,w,shape_key=token_key(L))
        ctx.save_for_backward(q,k,v,b,m,gate2,w,out2)
        return y.view_as(gate)
    @staticmethod
    def backward(ctx,dy):
        q,k,v,b,m,gate,w,out=ctx.saved_tensors
        dy=dy.reshape(-1,128).contiguous()
        dr,dg,a,delta=gate_backward(dy,w,gate,out)
        dq,dk,dv,db=attention_backward(q,k,v,b,m,delta,dr)
        dw=dy.T@a if ctx.needs_input_grad[-1] else None
        return dq,dk,dv,db,dg.view(1,q.shape[2],q.shape[2],128),dw

def forward(model,pair,mask=None):
    B,L,_,C=pair.shape
    q,k,v,g,b,residual=ln_backward.Front.apply(pair,model.ln_pair.weight,model.ln_pair.bias,model.ln_pair.eps,not model.starting,
        model.to_query.weight,model.to_key.weight,model.to_value.weight,model.to_gate.weight,model.to_bias.weight)
    q,k,v=(t.view(B,L,L,4,32).permute(0,3,1,2,4) for t in (q,k,v))
    b=b.permute(0,3,1,2)
    if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
    out=AttentionGate.apply(q,k,v,b,g,model.to_out.weight)
    if not model.starting:out=out.transpose(1,2).contiguous()
    if model.p_drop>0 and model.training:out=out*model._make_drop_scale(pair,model.p_drop)
    return residual+out
