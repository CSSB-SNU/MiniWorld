"""The pre-pass Front.backward, paired with frozen native baseline binaries."""
import torch
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward

class LegacyFront(torch.autograd.Function):
    forward=staticmethod(ln_backward.Front.forward)
    @staticmethod
    def backward(ctx,dq,dk,dv,dg,db,residual):
        x,z,mean,rstd,gamma,*weights=ctx.saved_tensors
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip((dq,dk,dv,dg,db),weights)]
        dx,dgamma,dbeta=ln_backward._backward(dy,weights,x,mean,rstd,gamma,residual,x.shape[1],ctx.ending)
        zz=z.reshape(-1,128)
        dw=[g.T@zz if needed else None for g,needed in zip(dy,ctx.needs_input_grad[5:])]
        return dx,dgamma,dbeta,None,None,*dw
