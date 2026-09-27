"""Separate warpgroups compute the exact packed and scalar LN orders."""
from pathlib import Path
import torch
from wide_dual_norm import DualNorm,DualNormBackward
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class ParallelDualNorm(DualNorm):
    def __init__(self,p,original):
        super().__init__(p,original)
        root=Path(__file__).resolve().parent
        body=(root/'wide_parallel_dual_norm.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0']
        self.cubin=T.compile_text(body,flags);self.threads=256;self.smem=65536+128+8192
        self.norm=T.load_unit(str(self.cubin),'mw_wide_parallel_dual_norm').kernel('mw_wide_parallel_dual_norm');self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),self.threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
