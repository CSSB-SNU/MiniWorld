from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class ChunkedAffineLN:
    def __init__(self,p,chunk=8):
        root=Path(__file__).resolve().parent;h=2*p.D;rows=16
        self.smem=2*rows*h*2+128+h*4+rows*8
        mb=3 if p.D==512 else 4
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        body=(root/'wide_chunked_affine_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DAFFINE_CHUNK={chunk}',f'-DMIN_BLOCKS={mb}']
        self.cubin=T.compile_text(body,flags)
        unit=T.load_unit(str(self.cubin),'mw_wide_chunked_affine_ln')
        self.kernel=unit.kernel('mw_wide_chunked_affine_ln');self.kernel.set_max_dynamic_smem(self.smem)
        self.reduce=unit.kernel('mw_wide_chunked_affine_reduce')
        drv=self.kernel.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),128,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        self.partial=torch.empty((self.grid,2*h),device=p.x.device,dtype=torch.float32);self.h=h
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.floats[5],p.floats[6],p.floats[2],self.partial,p.M])
        self.rp=L.Struct([self.partial,p.floats[10],p.floats[11],self.grid])
    def __call__(self):
        self.kernel.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)
        self.reduce.launch(((self.h+127)//128,1,1),(128,1,1),[self.rp],0)
