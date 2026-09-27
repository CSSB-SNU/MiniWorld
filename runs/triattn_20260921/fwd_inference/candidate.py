"""Explicit experimental inference entry. Load before capture/compilation; never builds."""
import torch
from torch.nn import functional as F
from miniworld_engine.kernels import _compile
from native import extension

SELECTED_ARTIFACTS = {384:'h4kv_local1',768:'hot6t',1024:'qfull4v2'}
SELECTED_OUTPUT_ARTIFACT = 'outproj_c1'
_DEFAULT_OUTPUT = object()


def load(artifact=None, fuse_front=True, front_artifact='front8', out_artifact=_DEFAULT_OUTPUT):
    # The default entry gets the qualified tail. Explicit core checkpoints retain
    # their historical cuBLAS tail unless output fusion is requested explicitly.
    if out_artifact is _DEFAULT_OUTPUT:
        out_artifact = SELECTED_OUTPUT_ARTIFACT if artifact is None else None
    if artifact is None:
        routes = {length:load(name,fuse_front,front_artifact,out_artifact)
                  for length,name in SELECTED_ARTIFACTS.items()}

        def selected(model,x,mask=None):
            if x.ndim!=4 or x.shape[1] not in routes:
                raise ValueError('Selected inference entry requires L384/768/1024.')
            return routes[x.shape[1]](model,x,mask)
        return selected
    ext = extension(artifact)
    head_last = hasattr(ext,'bias_head_last') and ext.bias_head_last()
    if head_last and front_artifact=='front8': front_artifact='front8_hlast'
    envelope = extension(front_artifact) if fuse_front else None
    out_ext = extension(out_artifact) if out_artifact else None

    def core_fake(z,wq,wk,wv,wg,bias): return torch.empty_like(z)

    @_compile.opaque(fake=core_fake,name='triattn_inference_core_'+artifact)
    def core(z:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,wv:torch.Tensor,wg:torch.Tensor,
             bias:torch.Tensor) -> torch.Tensor:
        return ext.forward(z,wq,wk,wv,wg,bias)

    def front_fake(x,gamma,beta,wb,mask,eps,ending):
        shape=(1,x.shape[1],x.shape[2],4) if head_last else (1,4,x.shape[1],x.shape[2])
        return torch.empty_like(x),torch.empty(shape,device=x.device,dtype=x.dtype)

    @_compile.opaque(fake=front_fake,name='triattn_inference_front_'+artifact)
    def front(x:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,wb:torch.Tensor,mask:torch.Tensor,
              eps:float,ending:bool) -> tuple[torch.Tensor,torch.Tensor]:
        return tuple(envelope.front(x,gamma,beta,wb,mask,eps,ending))

    def post_fake(x,out,ending): return torch.empty_like(x)

    @_compile.opaque(fake=post_fake,name='triattn_inference_post_'+artifact)
    def post(x:torch.Tensor,out:torch.Tensor,ending:bool) -> torch.Tensor:
        return envelope.post(x,out,ending)

    output = None
    if out_ext is not None:
        def output_fake(gated,weight,x,ending):return torch.empty_like(x)

        @_compile.opaque(fake=output_fake,name='triattn_inference_output_'+artifact+'_'+out_artifact)
        def output(gated:torch.Tensor,weight:torch.Tensor,x:torch.Tensor,ending:bool) -> torch.Tensor:
            return out_ext.forward(gated,weight,x,ending)

    fused_front_core = None
    if hasattr(ext,'front_kv'):
        def combined_fake(x,gamma,beta,wb,mask,eps,ending,wq,wk,wv,wg):return torch.empty_like(x)

        @_compile.opaque(fake=combined_fake,name='triattn_inference_front_core_'+artifact)
        def fused_front_core(x:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,wb:torch.Tensor,
                             mask:torch.Tensor,eps:float,ending:bool,wq:torch.Tensor,wk:torch.Tensor,
                             wv:torch.Tensor,wg:torch.Tensor) -> torch.Tensor:
            z,bias,key,value=ext.front_kv(x,gamma,beta,wb,mask,eps,ending,wk,wv)
            return ext.forward_kv(z,wq,wg,bias,key,value)

    def forward(model,x,mask=None):
        if torch.is_grad_enabled() or model.training:
            raise RuntimeError('This entry requires eval() and no_grad()/inference_mode().')
        if not (model.use_self_attention and not model.use_qk_norm and model.n_head==4):
            raise ValueError('Self attention with four D32 heads required.')
        if not (x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous() and
                x.ndim==4 and x.shape[0]==1 and x.shape[1]==x.shape[2] and x.shape[3]==128 and
                x.shape[1] in (384,768,1024)):
            raise ValueError('Unsupported inference input.')
        for name in ('to_query','to_key','to_value','to_gate','to_out'):
            w = getattr(model,name).weight
            if w.shape != (128,128) or w.dtype!=x.dtype or w.device!=x.device or not w.is_contiguous():
                raise ValueError('Contiguous BF16 128x128 projections on the input device required.')
        if fuse_front and fused_front_core is not None:
            mm = mask if mask is not None else torch.empty(0,device=x.device,dtype=torch.bool)
            gated = fused_front_core(x,model.ln_pair.weight,model.ln_pair.bias,model.to_bias.weight,mm,
                                    model.ln_pair.eps,not model.starting,model.to_query.weight,
                                    model.to_key.weight,model.to_value.weight,model.to_gate.weight)
            if output is not None:return output(gated,model.to_out.weight,x,not model.starting)
            return post(x,F.linear(gated,model.to_out.weight),not model.starting)
        if fuse_front:
            mm = mask if mask is not None else torch.empty(0,device=x.device,dtype=torch.bool)
            z,bias = front(x,model.ln_pair.weight,model.ln_pair.bias,model.to_bias.weight,mm,
                          model.ln_pair.eps,not model.starting)
        else:
            pair = x if model.starting else x.transpose(1,2).contiguous()
            z = model._layernorm(pair,model._backend)
            bias = F.linear(z,model.to_bias.weight)
            if not head_last: bias=bias.permute(0,3,1,2)
            if mask is not None:
                keep=mask[:,None,:,None] if head_last else mask[:,None,None,:]
                bias = bias.masked_fill(~keep,torch.finfo(bias.dtype).min)
        gated = core(z,model.to_query.weight,model.to_key.weight,model.to_value.weight,
                     model.to_gate.weight,bias.contiguous())
        if output is not None:return output(gated,model.to_out.weight,x,not model.starting)
        out = F.linear(gated,model.to_out.weight)
        if fuse_front: return post(x,out,not model.starting)
        if not model.starting: out = out.transpose(1,2).contiguous()
        return x+out
    return forward
