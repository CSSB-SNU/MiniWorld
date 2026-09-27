"""Explicit projection-dgrad fusion around the qualified wide attention core."""
import types
import torch
import candidate
from projection import Projections,extension
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.kernels.bias_only_attention import dispatch as bo_dispatch
from miniworld_engine.modules.functional import sigmoid_gate


def attach(model):
    assert model.use_self_attention and not model.use_qk_norm
    C=model.to_query.weight.shape[0]
    extension(C);candidate.preload(C//4)
    def attention(self,pair,mask=None):
        if not self.starting:pair=pair.transpose(1,2).contiguous()
        x=self._layernorm(pair,KernelBackend.TRITON)
        q,k,v,g,b=Projections.apply(x,self.to_query.weight,self.to_key.weight,self.to_value.weight,self.to_gate.weight,self.to_bias.weight)
        B,L,_,_=x.shape;D=C//4
        q,k,v=[t.view(B,L,L,4,D).permute(0,3,1,2,4) for t in (q,k,v)]
        b=b.permute(0,3,1,2)
        if mask is not None:b=b.masked_fill(~mask[:,None,None,:],torch.finfo(b.dtype).min)
        out=candidate.Attention.apply(q,k,v,b,'full')
        out=out.permute(0,2,3,1,4).reshape(B,L,L,C)
        if bo_dispatch.use_kernels(L):out=self._gate_out(g,out)
        else:out=self.to_out(sigmoid_gate(g,out))
        if not self.starting:out=out.transpose(1,2).contiguous()
        return out
    model._attention=types.MethodType(attention,model)
    return model
