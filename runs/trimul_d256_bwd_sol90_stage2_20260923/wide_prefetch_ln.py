"""Output-LN with narrow TMA row tiles and optional TMA dNorm loads."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class PrefetchLN:
    def __init__(self,p,rows=32,dn_tma=True,fence=False,minblocks=2,channel_affine=False):
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=128
        assert rows in (8,16,32)
        assert dn_tma and not channel_affine
        self.smem=max(4*rows*h*2+128,2*4*h*4)
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        if rows==8:helpers=(root/'tile_transpose8.cuh').read_text()
        body=(root/'wide_prefetch_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        body=body.replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        self.smem+=h*4
        flags=['-std=c++17' ,'-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),
               '-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}',
               f'-DLN_DN_TMA={int(dn_tma)}',f'-DLN_FENCE={int(fence)}',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        name='mw_wide_prefetch_ln'
        self.kernel=T.load_unit(str(self.cubin),name).kernel(name)
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='none' if rows==8 else '64B' if rows==32 else '32B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,128,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
