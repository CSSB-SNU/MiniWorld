from pathlib import Path
source=(Path(__file__).parent/'bench_baseline.py').read_text().split('\nq,k,v=')[0]
exec(compile(source,str(Path(__file__).parent/'bench_baseline.py'),'exec'))
import copy,types
import projections
from native import extension
ext=extension()

dy=[torch.randn(L*L,c,device='cuda',dtype=torch.bfloat16) for c in (128,128,128,128,4)]
w=[torch.randn(c,128,device='cuda',dtype=torch.bfloat16)*.1 for c in (128,128,128,128,4)]
ref=sum(g[:1024].double()@ww.double() for g,ww in zip(dy,w))
def baseline():
    result=dy[0]@w[0]
    for g,ww in zip(dy[1:],w[1:]):result=result+g@ww
    return result
for tile in (64,128):
    out=ext.dgrad(dy,w,tile)
    error=float((out[:1024].double()-ref).norm()/ref.norm())
    print('PROJECTION_ERROR',tile,error,flush=True)
    assert error<.002
    measure('projection_cuda%d'%tile,lambda:ext.dgrad(dy,w,tile))
measure('projection_baseline',baseline)
del dy,w,out,ref;gc.collect();torch.cuda.empty_cache()

for ending in (False,True):
    torch.manual_seed(92301)
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    base._fuse_projection_backward=False
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    dy=torch.randn_like(x);ref=None
    for name in ('baseline','addmm','cuda64','cuda128'):
        model=copy.deepcopy(base)
        if name!='baseline':
            projections.MODE=name
            model._attention=types.MethodType(projections.attention,model)
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
        measure('%s_backward_e%d'%(name,ending),lambda:torch.autograd.grad(y,params,dy,retain_graph=True),s)
        del y,params,model;gc.collect();torch.cuda.empty_cache()
    del base,x,dy,ref;gc.collect();torch.cuda.empty_cache()
report['complete']=True;save()
