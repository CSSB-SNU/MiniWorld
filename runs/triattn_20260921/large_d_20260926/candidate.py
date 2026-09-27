"""Explicit wide attention experiment, isolated from installed serving dispatch."""
from pathlib import Path
import sys
import types
import torch
from native import extension
from aux import extension as aux_extension

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'bwd/baseline_comparison_20260926'))
from original_triton import core, TriangleAttention as OriginalTriton
from miniworld_engine.autotune.shape_key import token_key


def projection_layout(x):
    L,D=x.shape[2],x.shape[4]
    if x.stride(4)==1 and x.stride(1)==D and x.stride(3)==4*D and x.stride(2)==L*4*D:
        return x
    return x.permute(0,2,3,1,4).contiguous().permute(0,3,1,2,4)


def preload(head_dim,mode='full'):
    if mode!='bwd':extension(head_dim,'fwd')
    if mode!='fwd':
        extension(head_dim,'dq');extension(head_dim,'dkdv');aux_extension()


def forward(q,k,v,b,mode='full'):
    if mode=='bwd':return core._tri_attn_fwd(q,k,v,b,token_key(q.shape[2]))
    return extension(q.shape[-1],'fwd').forward(q,k,v,b)


def backward(q,k,v,b,m,out,dy,mode='full'):
    L,D=q.shape[2],q.shape[4]
    dy=projection_layout(dy)
    if mode=='fwd':
        dq,dk,dv,db=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))
        return dq,dk,dv,db.reshape(1,4,L,L,L).sum(2)
    delta=aux_extension().delta(out,dy)
    ext=extension(D,'dkdv')
    dk,dv,db=ext.backward(q,k,v,b,m,delta,dy,ext.row_group)
    dq=extension(D,'dq').backward(q,k,v,b,m,delta,dy)
    return dq,dk,dv,db


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,b,mode):
        b=b.contiguous()
        out,m=forward(q,k,v,b,mode)
        ctx.save_for_backward(q,k,v,b,m,out);ctx.mode=mode
        return out

    @staticmethod
    def backward(ctx,dy):
        return (*backward(*ctx.saved_tensors,dy,ctx.mode),None)


def attach(model,mode='full'):
    assert model.n_head==4 and model.to_query.weight.shape[0] in (256,512)
    preload(model.to_query.weight.shape[0]//4,mode)
    def attention(self,q,k,v,b,backend):
        return Attention.apply(q,k,v,b,mode)
    model._kernel_triangle_attention=types.MethodType(attention,model)
    return model
