"""H100 native CUDA/TMA dK/dV with grouped shared-bias reduction.

Forward, delta and dQ retain the established implementation. Native outputs are
projection-layout BF16 gradients; the large per-row bias buffer is eliminated.
"""
from __future__ import annotations
import functools,importlib.util,json,sys
from pathlib import Path
import torch
import triton
from miniworld_engine.kernels._compile import device_constant,opaque
from miniworld_engine.autotune.shape_key import token_key,pack

from miniworld_engine.kernels.triangle_attention.cuda.bias_backward import _extension, can_use
from native import extension

def _fake(q,k,v,b,m,out,dy):
    B,H,L,_,D=q.shape
    def grad():return torch.empty((B,L,L,H*D),device=q.device,dtype=q.dtype).view(B,L,L,H,D).permute(0,3,1,2,4)
    return grad(),grad(),grad(),torch.empty_like(b)

@opaque(fake=_fake,name='triangle_attention_dq_native_experiment')
def _backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,b:torch.Tensor,m:torch.Tensor,out:torch.Tensor,dy:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    from miniworld_engine.kernels.triangle_attention.triton import main as core
    B,H,L,_,D=q.shape
    if dy.dtype!=q.dtype:dy=dy.to(q.dtype)
    if dy.stride(4)!=1 or dy.stride(1)!=D or dy.stride(3)!=H*D or dy.stride(2)!=L*H*D:
        dy=dy.permute(0,2,3,1,4).contiguous().view(B,L,L,H,D).permute(0,3,1,2,4)
    delta=torch.empty((B,H,L,L),device=q.device,dtype=torch.float32)
    pre_grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),H*L,1]
    core._attn_bwd_preprocess[pre_grid](out,dy,delta,*out.stride(),*dy.stride(),H*L,B,L,D,
        shape_key=pack(token_key(L),HEAD_DIM=D),HEAD_DIM_PAD=D)
    dk,dv,db=_extension().backward(q,k,v,b,m,delta,dy,4)
    dq=extension().backward(q,k,v,b,m,delta,dy)
    return dq,dk,dv,db

class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b):
        from miniworld_engine.kernels.triangle_attention.triton import main as core
        b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(q.shape[2]))
        ctx.save_for_backward(q,k,v,b,m,out)
        return out
    @staticmethod
    def backward(ctx,dy):return _backward(*ctx.saved_tensors,dy)

def attention(q,k,v,b):return _Attention.apply(q,k,v,b)
