"""CTA phase fractions from an instrumented control, with output equality."""
import argparse
import json
from pathlib import Path
import torch
from native import extension

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension('qkv_compact_phase');base=extension('qg_scoped')
report=dict(instrumented=True,performance_selection=False,records=[])
for L in (384,768,1024):
    torch.manual_seed(92799)
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    weights=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(4)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)
    b[...,::7]=torch.finfo(b.dtype).min
    for _ in range(5):out=ext.forward(z,*weights,b)
    torch.cuda.synchronize()
    ref=base.forward(z,*weights,b)
    assert all(torch.equal(x,y) for x,y in zip(out[:6],ref))
    stamps=out[-1].cpu()
    p=(stamps[...,1]-stamps[...,0]).double().flatten()
    attn=(stamps[...,2]-stamps[...,1]).double().flatten()
    assert (p>0).all() and (attn>0).all()
    qs=torch.tensor([0,.5,.9,1.],dtype=torch.double)
    row=dict(length=L,blocks=4*L,bitwise=True,
             projection_cycles=torch.quantile(p,qs).tolist(),
             attention_cycles=torch.quantile(attn,qs).tolist(),
             projection_fraction_mean=float((p/(p+attn)).mean()),
             total_cycles=torch.quantile(p+attn,qs).tolist())
    report['records'].append(row);print(row,flush=True)
report['complete']=True
a.output.write_text(json.dumps(report,indent=2)+'\n')
