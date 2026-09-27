from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class SpecializedOutputEpi:
    def __init__(self,plan,vec=1):
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1',f'-DWIDTH={plan.p.D}',f'-DLENGTH={plan.p.n}',f'-DVEC={vec}']
        self.cubin=T.compile(Path(__file__).with_suffix('.cu'),flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_specialized_output_epi').kernel('mw_wide_specialized_output_epi')
    def launch(self,*args):return self.kernel.launch(*args)
