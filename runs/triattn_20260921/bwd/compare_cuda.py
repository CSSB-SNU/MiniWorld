"""CUDA vendor control with the same Triton module surrounds and all gradients.

Comparison only: no installed dispatch changes. Includes layout conversions.
"""
from pathlib import Path
source=(Path(__file__).parent/'bench_baseline.py').read_text().split('\nq,k,v=')[0]
exec(compile(source,str(Path(__file__).parent/'bench_baseline.py'),'exec'))
import copy,types
from cuequivariance_torch import triangle_attention

def cuda_attention(self,q,k,v,b,backend):
    args=[t.permute(0,2,1,3,4).contiguous() for t in (q,k,v)]
    return triangle_attention(*args,b.unsqueeze(1).float()).permute(0,2,1,3,4)

for ending in (False,True):
    torch.manual_seed(92301)
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    base._fuse_projection_backward=False
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    cand=copy.deepcopy(base);cand._kernel_triangle_attention=types.MethodType(cuda_attention,cand)
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    dy=torch.randn_like(x);ref=None
    for name,model in [('triton',base),('cuda_vendor',cand)]:
        params=(x,*model.parameters());s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):y=model(x,mask)
        torch.cuda.current_stream().wait_stream(s)
        values=(y.detach(),*torch.autograd.grad(y,params,dy,retain_graph=True))
        assert all(torch.isfinite(v).all() for v in values)
        if ref is None:ref=[v.detach().clone() for v in values]
        errors=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(values,ref)]
        report.setdefault('initial_gradient_comparison',{})['%s_e%d'%(name,ending)]=dict(names=['output','input',*[n for n,p in model.named_parameters()]],relative_l2=errors)
        save();print('ERRORS',name,ending,errors,flush=True)
        assert max(errors)<.025,errors
        del values
        def backward():return torch.autograd.grad(y,params,dy,retain_graph=True)
        measure('%s_backward_e%d'%(name,ending),backward,s)
        def fwdbwd():return torch.autograd.grad(model(x,mask),params,dy)
        measure('%s_forward_backward_e%d'%(name,ending),fwdbwd)
        del y,params
    del base,cand,x,dy,ref;gc.collect();torch.cuda.empty_cache()
report['complete']=True;save()
