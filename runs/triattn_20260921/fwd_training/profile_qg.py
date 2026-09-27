"""Profile all projections plus attention, including training saves."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--length',type=int,default=768)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();L=a.length
ext=extension('q_only_head4' if a.artifact=='baseline' else a.artifact)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
w=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(4)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
b[...,::7]=torch.finfo(b.dtype).min
def run():
    if a.artifact=='baseline':
        gate=F.linear(z,w[3])
        return (*ext.forward(z,*w[:3],b),gate)
    return ext.forward(z,*w,b)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(5):out=run()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=stream):out=run()
for _ in range(30):g.replay()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
assert all(torch.isfinite(x).all() for x in out)
a.output.write_text(json.dumps(dict(artifact=a.artifact,length=L,complete=True),indent=2)+'\n')
