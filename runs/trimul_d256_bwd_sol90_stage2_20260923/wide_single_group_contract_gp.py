"""N64 accumulators and full TMA epilogue permit higher fused occupancy."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_contract_gp_tma import TmaContractGP
class SingleGroupContractGP(TmaContractGP):
    def __init__(self,plan,slots=2):
        super().__init__(plan)
        root=Path(__file__).resolve().parent;mb=4 if slots==2 else 3
        body=(root/'wide_single_group_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}',f'-DINPUT_SLOTS={slots}',f'-DMIN_BLOCKS={mb}',f'-DCHANNEL_GROUP={plan.channel_group}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_single_group_contract_gp').kernel('mw_wide_single_group_contract_gp')
        self.smem=24576*slots+128;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)*(self.p.n//64),1,1),(128,1,1),[self.params],self.smem)
