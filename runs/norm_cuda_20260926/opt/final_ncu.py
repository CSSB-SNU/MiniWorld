import argparse,torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,extension
from miniworld_engine.kernels.norm_cuda.wide import extension as wide_extension
p=argparse.ArgumentParser();p.add_argument('--case',type=int,default=0);a=p.parse_args()
extension();wide_extension();torch.set_num_threads(4)
m,d,rms=[(147456,128,False),(147456,384,True),(2048,4096,False)][a.case]
x=torch.randn(1,m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x);params=tuple(v for v in (x,w,b) if v is not None)
for _ in range(4):
 y=cuda_rmsnorm(x,w) if rms else cuda_layernorm(x,w,b)
 torch.autograd.grad(y,params,dy)
torch.cuda.synchronize()
