"""Warm-cache NCU measurement of the actual inference fused core, without training saves."""
import argparse
import json
from pathlib import Path
import torch
from native import extension

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--length',type=int,default=768)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();L=a.length
ext=extension(a.artifact)
torch.manual_seed(92675)
with torch.inference_mode():
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    w=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(4)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
    b[...,::7]=torch.finfo(b.dtype).min
    if hasattr(ext,'bias_head_last') and ext.bias_head_last():b=b.permute(0,2,3,1).contiguous()
    for _ in range(5):out=ext.forward(z,*w,b)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    out=ext.forward(z,*w,b)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert torch.isfinite(out).all()
a.output.write_text(json.dumps(dict(artifact=a.artifact,length=L,complete=True))+'\n')
