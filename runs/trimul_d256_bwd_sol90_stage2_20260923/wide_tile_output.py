"""Dense forward output with a narrower, more resident normalization tile."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from wide_dense_output import DenseOutput


class TileOutput(DenseOutput):
    def __init__(self,f,p,products,rows=32,threads=128):
        super().__init__(f,p,products,128 if p.D<512 else 256,packed_unroll=1 if p.D<512 else 16)
        self.epi_cubin=self.cubin
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=threads;self.smem=rows*h*2+128
        helpers=(root/'tile_transpose.cuh').read_text()
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),
               '-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DOUTPUT_ROWS={rows}',f'-DOUTPUT_THREADS={threads}']
        self.cubin=T.compile_text(body,flags)
        self.norm=T.load_unit(str(self.cubin),'mw_wide_tile_norm').kernel('mw_wide_tile_norm');self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='64B' if rows==32 else '32B',l2='128B')
        norm=L.tensor_map(p.tensors[6],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,norm,f.leaves[9],f.leaves[10],p.floats[5],p.floats[6],p.M])
