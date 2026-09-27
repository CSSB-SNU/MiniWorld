from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
R=Path(__file__).resolve().parent
class Pipe3:
 def __init__(self,old):
  self.old=old
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DWIDTH=256','-DGROUPS=4','-DKCHUNK=1','-DTMN_SIGMOID_TANH=1','-DMW_MINB=1','-DMW_K1_STREAM=0']
  out=T.compile(R/'pipe3.cu',flags);self.k=T.load_unit(str(out),'mw_d256_prepare_pipe3').kernel('mw_d256_prepare_pipe3');self.smem=65536+3*8192*5+640;self.k.set_max_dynamic_smem(self.smem)
 def __call__(self):self.k.launch((self.old.sms,1,1),(512,1,1),[self.old.params],self.smem)
