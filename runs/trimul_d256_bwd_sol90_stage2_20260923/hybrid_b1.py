"""TMA normalization/gradient kernels with dense GEMMs scheduled by cuBLAS."""
import torch,os
from blas_prepare import BlasPrepare
from ln import LN
class HybridB1:
 def __init__(self,p,leaves,use_cute=None):
  self.p=p;self.wp=leaves[6];self.prepare=BlasPrepare(p,leaves[6],leaves[5]);self.ln=LN(p)
  if use_cute if use_cute is not None else os.environ.get('B1_MODE')=='cute':
   from cute_prepare import CutePrepare
   self.prepare=CutePrepare(p,leaves[6],leaves[5])
  self.fused_dn=os.environ.get('DN_SLIM')=='1'
  if self.fused_dn:
   from dn_slim import DNSlim
   self.ln=DNSlim(p)
 def __call__(self):
  p=self.p;self.prepare()
  if not self.fused_dn:torch.mm(p.tensors[7],self.wp,out=p.tensors[9])
  torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
  torch.mm(p.dg.t(),p.xn.reshape(p.M,256),out=p.dwg)
  self.ln()
