"""Profile the isolated native O+LSE forward after warming a CUDA graph."""
import argparse
import json
from pathlib import Path
import torch
from native import extension

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--length',type=int,default=768)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args(); L=a.length
ext=extension(a.artifact)
q,k,v=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
       .view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(3)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
b[...,::7]=torch.finfo(b.dtype).min
stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(5): out=ext.forward(q,k,v,b)
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=stream): out=ext.forward(q,k,v,b)
for _ in range(30): g.replay()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
g.replay(); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
assert all(torch.isfinite(t).all() for t in out)
a.output.write_text(json.dumps(dict(artifact=a.artifact,length=L,complete=True,
    build=json.loads((Path(__file__).parent/a.artifact/'build-ready.json').read_text())),indent=2)+'\n')
