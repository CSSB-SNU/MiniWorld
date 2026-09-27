"""Profile one warmed replay, including delta, dQ and final bias reduction."""
import argparse,json
from pathlib import Path
import torch
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
import hybrid
from native import extension
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768);ap.add_argument('--rows',type=int,default=4);a=ap.parse_args()
L=a.length;torch.manual_seed(92301);extension()
q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5;b[...,::7]=torch.finfo(b.dtype).min
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
def run():
    if a.rows:return hybrid.backward(q,k,v,b,m,out,dy,a.rows)
    dq,dk,dv,db=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))
    return dq,dk,dv,db.reshape(1,4,L,L,L).sum(2)
s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):y=run()
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=s):y=run()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
assert all(torch.isfinite(x).all() for x in y)
