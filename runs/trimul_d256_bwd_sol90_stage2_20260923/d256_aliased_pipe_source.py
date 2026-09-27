"""Two compute warpgroups, aliased dL/dGate storage, two resident CTAs."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class AliasedPipeSource:
    def __init__(self,p,original,producer_regs=80):
        self.p=p;self.original=original;root=Path(__file__).resolve().parent
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={original.splits}',f'-DPRODUCER_REGS={producer_regs}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_aliased_pipe_source').kernel('mw_d256_aliased_pipe_source');self.smem=115072
        self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((32*self.original.splits,1,1),(256,1,1),[self.original.params],self.smem)
