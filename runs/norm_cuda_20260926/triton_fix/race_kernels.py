"""Racecheck the changed stores and selected schedules without timing sweeps/cuBLAS backward."""
import torch
import triton
from miniworld_engine.kernels.layernorm_linear.triton.fused import _lnl_fwd_kernel
from miniworld_engine.kernels.rmsnorm.triton.main import rmsnorm_fwd_kernel,rmsnorm_bwd_kernel

torch.set_num_threads(4)
torch.manual_seed(875)
for k,n,bk in [(128,16,128),(384,512,512),(137,17,64)]:
    m=35
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    g=torch.randn(k,device='cuda');b=torch.randn_like(g)
    w=torch.randn(n,k,device='cuda',dtype=x.dtype)
    y=torch.empty(m,n,device='cuda',dtype=x.dtype)
    mean=torch.empty(m,device='cuda');inv=torch.empty_like(mean)
    _lnl_fwd_kernel.fn[(triton.cdiv(m,32),)](x,w,x,g,b,y,m,n,k,1e-5,k,1,k,1,n,1,False,
        BLOCK_M1=32,BLOCK_N=32,BLOCK_K=bk,shape_key=0,
        Mean=mean,Rstd=inv,SAVE_STATS=True,num_warps=4,num_stages=1)
    expected=torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(),(k,),g,b,1e-5).to(x.dtype),w)
    assert (y.float()-expected.float()).norm()/expected.float().norm()<.004
    torch.testing.assert_close(mean,x.float().mean(-1),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(inv,torch.rsqrt(x.float().var(-1,unbiased=False)+1e-5),rtol=1e-5,atol=1e-6)
    print('LNL',k,n,bk,'PASS',flush=True)
for n,bm in [(128,64),(384,16),(1024,16)]:
    m=131
    x=torch.randn(m,n,device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
    dy=torch.randn_like(x);dx=torch.empty_like(x)
    w=torch.randn(n,device='cuda');dw=torch.zeros_like(w);inv=torch.empty(m,device='cuda')
    bk=triton.next_power_of_2(n);grid=(triton.cdiv(m,bm),)
    rmsnorm_fwd_kernel.fn[grid](x,y,w,inv,n,1,m,n,1e-5,bm,bk,0,True,num_warps=4,num_stages=1)
    rmsnorm_bwd_kernel.fn[grid](dx,dy,dw,x,w,inv,n,1,m,n,bm,bk,0,True,num_warps=4,num_stages=1)
    xf=x.float();rh=torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5);xh=xf*rh
    wd=dy.float()*w;ex=((wd-xh*(wd*xh).mean(-1,keepdim=True))*rh).to(x.dtype)
    ew=(dy.float()*xh).sum(0)
    for a,b in [(dx,ex),(dw,ew)]:assert (a.float()-b.float()).norm()/b.float().norm()<.004
    print('RMS',n,bm,'PASS',flush=True)
torch.cuda.synchronize()
