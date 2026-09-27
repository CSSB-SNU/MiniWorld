import argparse,json
from pathlib import Path
import torch
from native import extension
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=768);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
z=torch.randn(a.length*a.length,128,device='cuda',dtype=torch.bfloat16);dy=[torch.randn_like(z) for _ in range(4)]
ext=extension();split={384:2304,768:8960,1024:15936}[a.length]
graphs=[];outputs=[]
for fn in (lambda:[d.T@z for d in dy],lambda:ext.backward(dy,z,split)):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):out=fn()
 graphs.append(g);outputs.append(out)
for _ in range(3):
 for g in graphs:g.replay()
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
for g in graphs:g.replay()
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
a.output.write_text(json.dumps(dict(length=a.length,split=split,binary=ext.__file__))+'\n')
