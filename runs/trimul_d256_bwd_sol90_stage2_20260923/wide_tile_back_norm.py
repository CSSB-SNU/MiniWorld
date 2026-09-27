"""Tiled backward norm retaining the scalar-warp legacy D512 reduction."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class TileBackNorm:
    def __init__(self,p,rows=16,threads=128):
        root=Path(__file__).resolve().parent;d=p.D;h=2*d
        self.threads=threads;self.smem=rows*h*2+128
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('if constexpr(D<512)','if constexpr(true)').replace('mw_wide_tile_norm','mw_wide_tile_back_norm')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}',f'-DOUTPUT_ROWS={rows}',f'-DOUTPUT_THREADS={threads}']
        self.cubin=T.compile_text(body,flags)
        self.norm=T.load_unit(str(self.cubin),'mw_wide_tile_back_norm').kernel('mw_wide_tile_back_norm')
        self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='64B' if rows==32 else '32B',l2='128B')
        norm=L.tensor_map(p.tensors[6],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,norm,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])

    def __call__(self):self.norm.launch((self.grid,1,1),(self.threads,1,1),[self.np],self.smem)
