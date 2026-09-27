from pathlib import Path
import sys
import types
import torch
from torch.nn import functional as F
from build import extension
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;sys.path.insert(0,str(BASE))
import candidate
import projection
import accumulate
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.kernels.bias_only_attention import dispatch as bo_dispatch
from miniworld_engine.modules.functional import sigmoid_gate


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,compact):
        b=b.contiguous();o,m=candidate.forward(q,k,v,b)
        ctx.save_for_backward(q,k,v,b,m,o);ctx.compact=compact
        return o
    @staticmethod
    def backward(ctx,dy):
        q,k,v,b,m,o=ctx.saved_tensors
        if not ctx.compact:return (*candidate.backward(q,k,v,b,m,o,dy),None)
        dy=candidate.projection_layout(dy);D=q.shape[-1]
        delta=candidate.aux_extension().delta(o,dy)
        ext=extension('bias16',D*4)
        dk,dv,db=ext.backward(q,k,v,b,m,delta,dy,ext.row_group)
        dq=candidate.extension(D,'dq').backward(q,k,v,b,m,delta,dy)
        return dq,dk,dv,db,None


class Projections(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,wq,wk,wv,wg,wb,kind):
        weights=(wq,wk,wv,wg,wb);ctx.save_for_backward(x,*weights);ctx.kind=kind
        return tuple(F.linear(x,w) for w in weights)
    @staticmethod
    def backward(ctx,*grads):
        x,*weights=ctx.saved_tensors;C=x.shape[-1]
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip(grads,weights)]
        dx=None
        if ctx.needs_input_grad[0]:
            if ctx.kind.startswith('accumulate'):dx=accumulate.dgrad(dy,weights,ctx.kind).view_as(x)
            else:
                ext=projection.extension(C) if ctx.kind=='prior' else extension(ctx.kind,C)
                dx=ext.dgrad(dy,weights,64).view_as(x)
        xx=x.reshape(-1,C)
        dw=[g.T@xx if need else None for g,need in zip(dy,ctx.needs_input_grad[1:6])]
        return dx,*dw,None


def attach(model,kind=None,compact=True):
    C=model.to_query.weight.shape[0];assert C in (256,512) and model.n_head==4
    assert model.use_self_attention and not model.use_qk_norm
    candidate.preload(C//4)
    if compact:extension('bias16',C)
    if kind is None:
        def core(self,q,k,v,b,backend):return Attention.apply(q,k,v,b,compact)
        model._kernel_triangle_attention=types.MethodType(core,model)
        return model
    if kind=='prior':projection.extension(C)
    elif kind.startswith('accumulate'):extension(kind,512)
    else:extension(kind,C)
    def attention(self,pair,mask=None):
        if not self.starting:pair=pair.transpose(1,2).contiguous()
        x=self._layernorm(pair,KernelBackend.TRITON)
        q,k,v,g,b=Projections.apply(x,self.to_query.weight,self.to_key.weight,self.to_value.weight,self.to_gate.weight,self.to_bias.weight,kind)
        B,L,_,_=x.shape;D=C//4
        q,k,v=[t.view(B,L,L,4,D).permute(0,3,1,2,4) for t in (q,k,v)]
        b=b.permute(0,3,1,2)
        if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
        out=Attention.apply(q,k,v,b,compact).permute(0,2,3,1,4).reshape(B,L,L,C)
        out=self._gate_out(g,out) if bo_dispatch.use_kernels(L) else self.to_out(sigmoid_gate(g,out))
        if not self.starting:out=out.transpose(1,2).contiguous()
        return out
    model._attention=types.MethodType(attention,model);return model
