"""Diagnostic: vendor GEMM lower-cost schedule for the remaining B1 products."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent
class BlasB1:
 def __init__(self,p,prepare):
  self.p,self.prepare=p,prepare
  body=(R/'b1_ln_finish.cu').read_text().replace('#include "base.cuh"',(R/'base.cuh').read_text())
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWEIGHT_SPLITS=32','-DWIDTH=256']
  out=T.compile_text(body,flags);self.k=T.load_unit(str(out),'mw_d256_b1_ln').kernel('mw_d256_b1_ln');self.k.set_max_dynamic_smem(W.tuning(256)[2])
  self.wp=p.dwp # shape only; actual weight view is bound below by caller
 def __call__(self):
  p=self.p;b=self.prepare
  b.prepare.launch((b.sms,1,1),(128*b.groups,1,1),[b.params],b.smem)
  torch.mm(p.tensors[7],self.wp,out=p.tensors[9])
  torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
  torch.mm(p.dg.t(),p.xn.reshape(p.M,256),out=p.dwg)
  W.launch(self.k,p.params,132,D=256)
