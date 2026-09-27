from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
import torch
from opt_core.kernels import triattn_surround_tma as native
from build import build
ext=build();torch.manual_seed(2020);L=384
with torch.no_grad():
 z=torch.randn((L,L,128),device='cuda',dtype=torch.bfloat16)
 w=torch.randn((512,128),device='cuda',dtype=torch.bfloat16)/128**.5
 wb=torch.zeros((16,128),device='cuda',dtype=torch.bfloat16);wb[:4].normal_()
 lnw=torch.ones(128,device='cuda',dtype=torch.bfloat16);lnb=torch.zeros_like(lnw)
 def outputs():return [torch.empty((L,4,L,32),device='cuda',dtype=torch.bfloat16) for _ in range(3)]+[torch.empty_like(z),torch.empty((4,L,L),device='cuda',dtype=torch.float32)]
 for ending in (False,True):
  expected=outputs();actual=outputs()
  native._load().prologue(z,w,wb,lnw,lnb,*expected,1e-5,ending)
  ext.async2(z,w,wb,lnw,lnb,*actual,1e-5,ending)
  torch.cuda.synchronize();assert all(torch.equal(t,r) for t,r in zip(actual,expected)),ending
 print('PASS: independent consumer groups, both directions, exact Q/K/V/gate/bias',flush=True)
