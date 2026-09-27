"""128x128 contraction with three input slots and a TMA issuer warpgroup."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_contract_gp_tma import TmaContractGP

class PipeContractGP(TmaContractGP):
    def __init__(self,plan,fused=True,transposed=True):
        super().__init__(plan,fused,transposed)
        root=Path(__file__).resolve().parent
        body=(root/'wide_pipe_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_pipe_contract_gp').kernel('mw_wide_pipe_contract_gp')
        self.smem=98304+128;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(384,1,1),[self.params],self.smem)
