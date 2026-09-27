"""Experimental autograd-preserving projections; forward is unchanged."""
import torch
from torch.nn import functional as F
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.bias_only_attention import dispatch
from native import extension

MODE='cuda64'
def _fake(dy,w,tile):return torch.empty((dy[0].numel()//128,128),device=dy[0].device,dtype=dy[0].dtype)
@opaque(fake=_fake,name='triattn_projection_dgrad_experiment')
def _cuda(dy:list[torch.Tensor],w:list[torch.Tensor],tile:int)->torch.Tensor:
    return extension().dgrad(dy,w,tile)

class Projections(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,wq,wk,wv,wg,wb):
        ctx.save_for_backward(x,wq,wk,wv,wg,wb)
        ctx.mode=MODE
        return tuple(F.linear(x,w) for w in (wq,wk,wv,wg,wb))
    @staticmethod
    def backward(ctx,*grad):
        x,*weights=ctx.saved_tensors
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip(grad,weights)]
        if ctx.mode.startswith('cuda'):
            dx=_cuda(dy,weights,int(ctx.mode[4:]))
        else:
            dx=dy[0]@weights[0]
            for g,w in zip(dy[1:],weights[1:]):torch.addmm(dx,g,w,out=dx)
        xx=x.reshape(-1,128)
        dw=[g.T@xx for g in dy]
        return dx.reshape_as(x),*dw

def attention(self,pair,mask=None):
    # Isolated comparison for B1 C128 H4 D32 self-attention without QK norm.
    assert self.use_self_attention and not self.use_qk_norm
    if not self.starting:pair=pair.transpose(1,2).contiguous()
    pair=self._layernorm(pair,self._backend)
    q,k,v,g,b=Projections.apply(pair,self.to_query.weight,self.to_key.weight,self.to_value.weight,self.to_gate.weight,self.to_bias.weight)
    B,L,_,_=pair.shape
    q,k,v=[z.view(B,L,L,4,32).permute(0,3,1,2,4) for z in (q,k,v)]
    b=b.permute(0,3,1,2)
    if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
    out=self._kernel_triangle_attention(q,k,v,b,self._backend)
    out=out.permute(0,2,3,1,4).reshape(B,L,L,128)
    out=self._gate_out(g,out) if dispatch.use_kernels(L) else self.to_out(torch.sigmoid(g)*out)
    if not self.starting:out=out.transpose(1,2).contiguous()
    return out
