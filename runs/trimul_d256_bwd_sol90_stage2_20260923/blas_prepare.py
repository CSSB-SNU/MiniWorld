from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
R=Path(__file__).resolve().parent
class BlasPrepare:
 def __init__(self,p,wp,wg):
  self.p=p;self.wp=wp.t();self.wg=wg.t();self.proj=torch.empty_like(p.dg);self.gate=torch.empty_like(p.dg)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DTMN_SIGMOID_TANH=1','-DMW_MINB=1','-DMW_K1_STREAM=0']
  out=T.compile(R/'norm.cu',flags);unit=T.load_unit(str(out),'mw_d256_norm');self.norm=unit.kernel('mw_d256_norm');self.norm.set_max_dynamic_smem(65664);self.epi=unit.kernel('mw_d256_gate_grad')
  L=T._launch_module();tri=F.tm(p.tri,[64,64],[p.M,512],[p.M*2])
  self.np=L.Struct([tri,p.maps[3],p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M]);self.ep=L.Struct([self.proj,self.gate,p.dy,p.ds,p.tensors[7],p.dg,p.M*256,p.n*256])
 def normalize(self):self.norm.launch((264,1,1),(256,1,1),[self.np],65664)
 def products(self):
  torch.mm(self.p.tensors[6],self.wp,out=self.proj);torch.mm(self.p.xn.reshape(self.p.M,256),self.wg,out=self.gate)
 def epilogue(self):self.epi.launch((528,1,1),(256,1,1),[self.ep],0)
 def __call__(self):self.normalize();self.products();self.epilogue()
