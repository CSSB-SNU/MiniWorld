from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
import torch
from opt_core.kernels import triattn_surround_tma as native
from build_epi import build
ext=build();torch.manual_seed(2019)
L=384
with torch.no_grad():
 o=torch.randn((L,4,L,32),device='cuda',dtype=torch.bfloat16)
 g=torch.randn((L,L,128),device='cuda',dtype=torch.bfloat16)
 w=torch.randn((128,128),device='cuda',dtype=torch.bfloat16)/128**.5
 z=torch.randn_like(g);lookup=native._sigmoid_table(z.device)
 for ending in (False,True):
  expected=torch.empty_like(z);actual=torch.empty_like(z)
  native._load().epilogue(o,g,w,z,expected,lookup,ending)
  ext.split(o,g,w,z,actual,lookup,ending)
  torch.cuda.synchronize()
  bad=expected!=actual
  print('difference',ending,int(bad.sum()),float((expected.float()-actual.float()).abs().max()),expected[bad][:20],actual[bad][:20],flush=True)
  assert torch.equal(expected,actual),ending
  inplace=z.clone();ext.split(o,g,w,inplace,inplace,lookup,ending)
  torch.cuda.synchronize();assert torch.equal(expected,inplace),(ending,'inplace')
 print('PASS: persistent pipeline, both directions, separate and aliased residual output',flush=True)
