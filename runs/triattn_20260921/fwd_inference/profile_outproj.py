"""Profile the same warmed, out-of-place output boundary before and after fusion."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension

p=argparse.ArgumentParser();p.add_argument('--artifact',required=True)
p.add_argument('--length',type=int,required=True);p.add_argument('--ending',action='store_true')
p.add_argument('--output',type=Path,required=True);a=p.parse_args();L=a.length
ext=extension(a.artifact);front=extension('front8')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.manual_seed(92685)
with torch.inference_mode():
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    x=torch.randn_like(z);w=torch.randn(128,128,device='cuda',dtype=z.dtype)/128**.5
    funcs=[lambda:front.post(x,F.linear(z,w),a.ending),lambda:ext.forward(z,w,x,a.ending)]
    for fn in funcs:
        for _ in range(5):out=fn()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    outputs=[]
    for fn in funcs:outputs.append(fn());torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert torch.equal(*outputs)
a.output.write_text(json.dumps(dict(length=L,artifact=a.artifact,ending=a.ending,complete=True))+'\n')
