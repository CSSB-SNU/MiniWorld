"""Isolated compile-aware API. Not installed in engine dispatch until qualified."""
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
import hybrid

def _fake(q,k,v,b,m,out,dy,rows):
    B,H,L,_,D=q.shape
    def grad():
        return torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
    return grad(),grad(),grad(),torch.empty_like(b)

@opaque(fake=_fake,name='triangle_attention_grouped_bwd_experiment')
def _backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,b:torch.Tensor,m:torch.Tensor,out:torch.Tensor,dy:torch.Tensor,rows:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    B,H,L,_,D=q.shape
    if dy.dtype!=q.dtype:dy=dy.to(q.dtype)
    if dy.stride(4)!=1 or dy.stride(1)!=D or dy.stride(3)!=H*D or dy.stride(2)!=L*H*D:
        dy=dy.permute(0,2,3,1,4).contiguous().view(B,L,L,H,D).permute(0,3,1,2,4)
    return hybrid.backward(q,k,v,b,m,out,dy,rows)

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,rows):
        b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(q.shape[2]))
        ctx.save_for_backward(q,k,v,b,m,out);ctx.rows=rows
        return out
    @staticmethod
    def backward(ctx,dy):
        return (*_backward(*ctx.saved_tensors,dy,ctx.rows),None)
