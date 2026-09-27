"""Native LN/residual fusion vs installed projection + stock LN + residual."""
import argparse,json,os,statistics
from pathlib import Path
import torch
from native import extension
from miniworld_engine.kernels.triangle_attention import cuda as projection
from miniworld_engine.kernels.layernorm import compile_native as ln
from miniworld_engine.autotune.shape_key import both_key
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;M=L*L;torch.manual_seed(9334)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension()
x=torch.randn(M,128,device='cuda',dtype=torch.bfloat16)
gamma=torch.randn(128,device='cuda',dtype=torch.float32)*.1+1
beta=torch.randn_like(gamma)*.1
z,mean,rstd=ln._dispatch_fwd(x.view(1,L,L,128),gamma,beta,1e-5)
w=[torch.randn(c,128,device='cuda',dtype=torch.bfloat16)*.08 for c in (128,128,128,128,4)]
dy=[torch.randn(M,c,device='cuda',dtype=torch.bfloat16) for c in (128,128,128,128,4)]
residual=torch.randn_like(x)
result=dict(length=L,records=[])
def run():return ext.backward(dy,w,x,mean,rstd,gamma,residual,L,False)
st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
 for _ in range(3):y=run()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=st):y=run()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart();g.replay();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
