import argparse,json,os
from pathlib import Path
import torch,triton
from miniworld_engine.kernels.adaln.triton.inference import _adaln_gemm_gate_kernel
p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=128);p.add_argument('--warps',type=int,default=4);p.add_argument('--stages',type=int,default=1);p.add_argument('--rows',type=int,default=8192);p.add_argument('--block-n',type=int,default=256);p.add_argument('--block-m',type=int,default=128);p.add_argument('--output',required=True);a=p.parse_args()
torch.manual_seed(51);torch.backends.cuda.matmul.allow_tf32=False
m,n,k=a.rows,a.width,min(a.width,384)
x=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
c=torch.randn(m,k,device='cuda',dtype=x.dtype)
sw=torch.randn(k,n,device='cuda',dtype=x.dtype)/k**.5
bw=torch.randn_like(sw)/k**.5
sb=torch.randn(n,device='cuda',dtype=x.dtype)
r=torch.rsqrt(x.float().var(-1,unbiased=False)+1e-5);c1=x.float().mean(-1)*r
y=torch.empty_like(x)
fn=_adaln_gemm_gate_kernel.fn
launcher=fn[(triton.cdiv(m,a.block_m)*triton.cdiv(n,a.block_n),)](c,sw,sb,bw,x,r,c1,y,m,n,k,c.stride(0),sw.stride(0),bw.stride(0),x.stride(0),y.stride(0),BLOCK_M=a.block_m,BLOCK_N=a.block_n,BLOCK_K=32,GROUP_M=1,shape_key=0,num_warps=a.warps,num_stages=a.stages)
compiled=launcher
# Write assembly even if synchronization exposes a poisoned CUDA context.
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
for ext in ('ptx','ttgir'):
 out.with_suffix('.'+ext).write_text(compiled.asm[ext])
torch.cuda.synchronize()
ref=(c.float()@sw.float()+sb.float()).sigmoid()*(x.float()*r[:,None]-c1[:,None])+c.float()@bw.float()
err=float((y.float()-ref).norm()/ref.norm());assert err<.014,err
out.write_text(json.dumps({'args':vars(a),'relative_l2':err,'registers':compiled.n_regs,'spills':compiled.n_spills,'shared':compiled.metadata.shared},indent=2))
print(out.read_text(),flush=True)
