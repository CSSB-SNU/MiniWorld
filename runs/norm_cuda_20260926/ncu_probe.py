import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,extension
extension();torch.set_num_threads(4)
x=torch.randn(1,147456,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
w=torch.randn(128,device='cuda',requires_grad=True);b=torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x)
for _ in range(4):
 y=cuda_layernorm(x,w,b,threads=128,rows=64);torch.autograd.grad(y,(x,w,b),dy)
torch.cuda.synchronize()
