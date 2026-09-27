import argparse, json, itertools, time
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm, cuda_rmsnorm, cuda_layernorm_linear, extension

p=argparse.ArgumentParser();p.add_argument('--part',type=int,default=0);p.add_argument('--sanitize',action='store_true');a=p.parse_args()
torch.manual_seed(917+a.part); torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
extension()

def ref(x,w,b,eps,rms):
    acc=torch.float64 if x.dtype==torch.float64 else torch.float32
    xf=x.to(acc)
    if rms:
        y=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)
        if w is not None:y=y*w.to(acc)
    else:y=torch.nn.functional.layer_norm(xf,(x.shape[-1],),None if w is None else w.to(acc),None if b is None else b.to(acc),eps)
    return y.to(x.dtype)

cases=list(itertools.product([1,3,31,32,33,63,64,127,128,256,384,512,768,1024,4096,8193], [torch.float16,torch.bfloat16,torch.float32,torch.float64], [False,True]))
if a.sanitize:cases=[(33,torch.float32,False),(128,torch.bfloat16,False),(4096,torch.float16,True),(8193,torch.float64,False)]
results=[]
for i,(d,dt,rms) in enumerate(cases):
    if not a.sanitize and i%2!=a.part:continue
    for layout in range(3):
        m=[1,17,129][layout]
        raw=torch.randn(m,d*2,device='cuda',dtype=dt)
        x=(raw[:,:d] if layout==0 else raw[:,::2] if layout==1 else raw.t().contiguous().t()[:,:d]).detach().requires_grad_()
        pd=dt if i%3==0 else (torch.float64 if dt==torch.float64 else torch.float32)
        w=torch.randn(d,device='cuda',dtype=pd,requires_grad=True) if i%5 else None
        b=torch.randn(d,device='cuda',dtype=pd,requires_grad=True) if not rms and i%3 else None
        eps=1e-5
        fn=cuda_rmsnorm if rms else cuda_layernorm
        y=fn(x,w,eps=eps) if rms else fn(x,w,b,eps)
        yref=ref(x,w,b,eps,rms)
        tol={torch.float16:2e-3,torch.bfloat16:1.7e-2,torch.float32:3e-5,torch.float64:3e-11}[dt]
        torch.testing.assert_close(y,yref,atol=tol,rtol=tol)
        dy=torch.randn_like(y)
        params=[v for v in (x,w,b) if v is not None]
        grads=torch.autograd.grad(y,params,dy,retain_graph=True)
        refs=torch.autograd.grad(yref,params,dy)
        for g,r in zip(grads,refs):torch.testing.assert_close(g,r,atol=tol*max(1,m**.5),rtol=tol*2)
        # Exercise graph replay with changed inputs, not just fixed capture input.
        if layout==0 and d in (33,128,8193):
            xx=x.detach().clone().requires_grad_();ww=None if w is None else w.detach().clone().requires_grad_();bb=None if b is None else b.detach().clone().requires_grad_()
            def operation():
                yy=fn(xx,ww,eps=eps) if rms else fn(xx,ww,bb,eps)
                gg=torch.autograd.grad(yy,[v for v in (xx,ww,bb) if v is not None],dy)
                return yy,gg
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):operation()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out,gg=operation()
            with torch.no_grad():xx.copy_(torch.randn_like(xx))
            graph.replay();expect=ref(xx,ww,bb,eps,rms)
            eg=torch.autograd.grad(expect,[v for v in (xx,ww,bb) if v is not None],dy)
            torch.testing.assert_close(out,expect,atol=tol,rtol=tol)
            for g,r in zip(gg,eg):torch.testing.assert_close(g,r,atol=tol*max(1,m**.5),rtol=tol*2)
        results.append({'D':d,'dtype':str(dt),'rms':rms,'layout':layout,'ok':True})
# LayerNormLinear, arbitrary leading dims, non-aligned K/N, all parameter gradients.
for dt,d,n in itertools.product([torch.float16,torch.bfloat16,torch.float32,torch.float64],[33,128,384],[17,128]):
    if a.sanitize and (d,n)!=(33,17):continue
    x=torch.randn(2,9,d,device='cuda',dtype=dt,requires_grad=True)
    pd=torch.float64 if dt==torch.float64 else torch.float32
    w=torch.randn(d,device='cuda',dtype=pd,requires_grad=True);b=torch.randn_like(w,requires_grad=True)
    mat=(torch.randn(n,d,device='cuda',dtype=dt)/d**.5).requires_grad_();bias=torch.randn(n,device='cuda',dtype=dt,requires_grad=True)
    out=cuda_layernorm_linear(x,w,b,mat,bias);expected=torch.nn.functional.linear(ref(x,w,b,1e-5,False),mat,bias)
    tol={torch.float16:4e-3,torch.bfloat16:3e-2,torch.float32:6e-5,torch.float64:3e-11}[dt]
    torch.testing.assert_close(out,expected,atol=tol,rtol=tol)
    dy=torch.randn_like(out);params=[x,w,b,mat,bias]
    for g,r in zip(torch.autograd.grad(out,params,dy),torch.autograd.grad(expected,params,dy)):
        torch.testing.assert_close(g,r,atol=tol*6,rtol=tol*3)
    results.append({'op':'lnlinear','D':d,'N':n,'dtype':str(dt),'ok':True})
# Empty leading dimensions and input-dtype epsilon.
for fn in (cuda_layernorm,cuda_rmsnorm):
    x=torch.empty(0,33,device='cuda',requires_grad=True);w=torch.ones(33,device='cuda',requires_grad=True)
    out=fn(x,w);gx,gw=torch.autograd.grad(out,(x,w),torch.empty_like(out));assert gx.numel()==0 and not gw.count_nonzero()
x=torch.randn(17,33,device='cuda',dtype=torch.bfloat16)
torch.testing.assert_close(cuda_rmsnorm(x,eps=None),ref(x,None,None,torch.finfo(x.dtype).eps,True),atol=.016,rtol=.016)
torch.cuda.synchronize()
path=Path(__file__).parent/f'correctness-{a.part}{"-sanitize" if a.sanitize else ""}.json';path.write_text(json.dumps(results,indent=2));print('PASS',len(results),flush=True)
