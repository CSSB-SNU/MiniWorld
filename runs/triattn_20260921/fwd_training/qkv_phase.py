"""Diagnostic CTA phase distribution, in clock cycles (not selection timing)."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension
ext=extension('resident_qkv_phase');base=extension('cooperative_head2')
records=[]
for L in (384,768,1024):
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    w=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(3)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
    for _ in range(10):got=ext.forward(z,*w,b)
    torch.cuda.synchronize()
    qkv=[F.linear(z,t).view(1,L,L,4,32).permute(0,3,1,2,4) for t in w]
    ref=(*base.forward(*qkv,b),*qkv)
    assert all(torch.equal(x,y) for x,y in zip(got[:5],ref))
    trace=got[5].cpu().double();pro=trace[:,1]-trace[:,0]
    end=trace[:,2:].max(-1).values;attn=end-trace[:,1];total=end-trace[:,0]
    rec=dict(length=L,projection_cycles_quantiles=torch.quantile(pro,torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),
             attention_cycles_quantiles=torch.quantile(attn,torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),
             projection_fraction_median=(pro/total).median().item())
    records.append(rec);print('PHASE',rec,flush=True)
import os
out=Path(__file__).parent/('qkv-phase-'+os.environ['SLURM_JOB_ID']+'.json')
out.write_text(json.dumps(dict(diagnostic_only=True,records=records),indent=2)+'\n')
