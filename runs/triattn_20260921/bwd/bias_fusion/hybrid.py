import os
import torch
import triton
from miniworld_engine.autotune.shape_key import token_key,pack
from miniworld_engine.kernels.triangle_attention.triton import main as core
from native import extension

def delta(out,dy):
    _,H,L,_,D=out.shape
    value=torch.empty((1,H,L,L),device=out.device,dtype=torch.float32)
    grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),H*L,1]
    core._attn_bwd_preprocess[grid](out,dy,value,*out.stride(),*dy.stride(),H*L,1,L,D,
        shape_key=pack(token_key(L),HEAD_DIM=D),HEAD_DIM_PAD=D)
    return value

def backward(q,k,v,b,m,out,dy,rows=4):
    _,H,L,_,D=q.shape
    d=delta(out,dy)
    dk,dv,db=extension().backward(q,k,v,b,m,d,dy,rows)
    if os.environ.get('FUSION_NATIVE_DQ','0')=='1':
        from miniworld_engine.kernels.triangle_attention.cuda import dq_backward
        return dq_backward.backward(q,k,v,b,m,d,dy),dk,dv,db
    dq=torch.empty((1,L,L,H*D),device=q.device,dtype=q.dtype).view(1,L,L,H,D).permute(0,3,1,2,4)
    grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),1,H*L]
    core._attn_bwd_dq[grid](q,k,v,b,D**-.5,dy,dq,m,d,*q.stride(),*dq.stride(),*dy.stride(),*b.stride(),
        L,H*L,D,HEAD_DIM_PAD=D,shape_key=pack(token_key(L),HEAD_DIM=D))
    return dq,dk,dv,db

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,rows):
        b=b.contiguous()
        out,m=core._tri_attn_fwd(q,k,v,b,token_key(q.shape[2]))
        ctx.save_for_backward(q,k,v,b,m,out);ctx.rows=rows
        return out
    @staticmethod
    def backward(ctx,dy):
        q,k,v,b,m,out=ctx.saved_tensors
        if dy.dtype!=q.dtype:dy=dy.to(q.dtype)
        if dy.stride()!=q.stride():
            B,H,L,_,D=q.shape
            dy=dy.permute(0,2,3,1,4).contiguous().view(B,L,L,H,D).permute(0,3,1,2,4)
        return (*backward(q,k,v,b,m,out,dy,ctx.rows),None)
