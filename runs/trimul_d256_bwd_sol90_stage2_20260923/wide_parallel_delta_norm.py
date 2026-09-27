"""Compute the two exact statistic orders in parallel; publish one norm and patches."""
from pathlib import Path
import torch
from wide_delta_norm import DeltaNorm
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class ParallelDeltaNorm(DeltaNorm):
    def __init__(self,p,original,minblocks=2):
        super().__init__(p,original,cached=True,unswitch=True)
        root=Path(__file__).resolve().parent
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        body=(root/'wide_parallel_delta_norm.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        pack_regs,scalar_regs=(160,96) if minblocks==2 else (104,56)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DDELTA_MINBLOCKS={minblocks}',f'-DDELTA_PACKED_REGS={pack_regs}',f'-DDELTA_SCALAR_REGS={scalar_regs}']
        self.cubin=T.compile_text(body,flags)
        self.native_norm=T.load_unit(str(self.cubin),'mw_wide_parallel_delta_norm').kernel('mw_wide_parallel_delta_norm')
        self.smem=16*1024*2+128+1024*8+128;self.threads=256;self.native_norm.set_max_dynamic_smem(self.smem)
        drv=self.native_norm.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.native_norm.handle)),256,self.smem)))
        assert self.occupancy>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
