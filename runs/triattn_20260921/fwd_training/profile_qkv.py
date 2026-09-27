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
ext=extension('cooperative_head2' if a.artifact=='baseline' else a.artifact)
z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
w=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(3)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
b[...,::7]=torch.finfo(b.dtype).min
def run():
    if a.artifact=='baseline':
        q,k,v=[F.linear(z,t).view(1,L,L,4,32).permute(0,3,1,2,4) for t in w]
        return (*ext.forward(q,k,v,b),q,k,v)
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
