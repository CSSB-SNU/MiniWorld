import torch,json
from pathlib import Path
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,cuda_layernorm_linear,extension
extension();torch.manual_seed(725);torch.set_num_threads(4)
results=[]
for rms in (False,True):
    fn=cuda_rmsnorm if rms else cuda_layernorm
    x=torch.randn(3,7,device='cuda',dtype=torch.float64,requires_grad=True)
    w=torch.randn(7,device='cuda',dtype=torch.float64,requires_grad=True)
    b=torch.randn(7,device='cuda',dtype=torch.float64,requires_grad=True)
    run=(lambda x,w:fn(x,w)) if rms else (lambda x,w,b:fn(x,w,b))
    args=(x,w) if rms else (x,w,b)
    assert torch.autograd.gradcheck(run,args,eps=1e-6,atol=2e-6,rtol=2e-5)
    results.append({'gradcheck':rms,'ok':True})
    compiled=torch.compile(run,fullgraph=True)
    for _ in range(2):
        with torch.no_grad():x.copy_(torch.randn_like(x));w.copy_(torch.randn_like(w))
        out=compiled(*args);expected=run(*args)
        torch.testing.assert_close(out,expected)
        dy=torch.randn_like(out)
        for g,r in zip(torch.autograd.grad(out,args,dy),torch.autograd.grad(expected,args,dy)):
            torch.testing.assert_close(g,r)
    results.append({'compile_fullgraph':rms,'ok':True})
# FP32 large offsets, constants, low variance, and low-precision epsilon.
for rms in (False,True):
    for dt in (torch.float16,torch.bfloat16,torch.float32):
        for scale,offset in [(0,0),(0,32),(1e-4,0),(1,1000)]:
            x=(torch.randn(31,384,device='cuda')*scale+offset).to(dt).requires_grad_()
            w=torch.randn(384,device='cuda',requires_grad=True)
            y=cuda_rmsnorm(x,w) if rms else cuda_layernorm(x,w)
            xf=x.float()
            ref=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*w).to(dt) if rms else torch.nn.functional.layer_norm(xf,(384,),w,eps=1e-5).to(dt)
            tol=.03 if dt==torch.bfloat16 else .004 if dt==torch.float16 else .0008
            torch.testing.assert_close(y,ref,atol=tol,rtol=tol)
            dy=torch.randn_like(y)
            for g,r in zip(torch.autograd.grad(y,(x,w),dy),torch.autograd.grad(ref,(x,w),dy)):
                torch.testing.assert_close(g,r,atol=tol*10,rtol=tol*3)
            results.append({'stress':str(dt),'rms':rms,'scale':scale,'offset':offset,'ok':True})
Path(__file__).with_name('qualification.json').write_text(json.dumps(results,indent=2));print('QUALIFICATION PASS',len(results),flush=True)
