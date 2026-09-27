"""Prototype integration; switch Front only before capture, never during replay."""
import importlib.util
import sys
from pathlib import Path
import torch
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward as installed
spec=importlib.util.spec_from_file_location('candidate_wgrad',Path(__file__).parent/'package/wgrad_backward.py')
wgrad=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wgrad;spec.loader.exec_module(wgrad)
BaselineFront=installed.Front

class CandidateFront(torch.autograd.Function):
    forward=staticmethod(BaselineFront.forward)
    @staticmethod
    def backward(ctx,dq,dk,dv,dg,db,residual):
        x,z,mean,rstd,gamma,*weights=ctx.saved_tensors
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip((dq,dk,dv,dg,db),weights)]
        dx,dgamma,dbeta=installed._backward(dy,weights,x,mean,rstd,gamma,residual,x.shape[1],ctx.ending)
        zz=z.reshape(-1,128)
        needed=ctx.needs_input_grad[5:]
        if all(needed[:4]) and wgrad.can_use(dy[:4],zz):
            dw=[*wgrad.backward(dy[:4],zz),dy[4].T@zz if needed[4] else None]
        else:
            dw=[g.T@zz if n else None for g,n in zip(dy,needed)]
        return dx,dgamma,dbeta,None,None,*dw

def select(candidate):
    installed.Front=CandidateFront if candidate else BaselineFront
