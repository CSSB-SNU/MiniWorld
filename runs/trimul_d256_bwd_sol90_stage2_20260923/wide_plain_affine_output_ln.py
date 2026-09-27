"""Independent affine worker and two protected input slots for output LN."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class PlainAffineOutputLN:
    def __init__(self,p):
        root=Path(__file__).resolve().parent;h=2*p.D;rows=16
        self.smem=rows*h*6+128+h*4
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        body=(root/'wide_affine_output_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        self.p=p
        body=body.replace(' for(int c=blockIdx.x*256+tid;c<H;c+=gridDim.x*256){p.dg[c]=0;p.db[c]=0;}','')
        body=body.replace('__syncthreads();cooperative_groups::this_grid().sync();','__syncthreads();')
        body=body.replace('mw_wide_affine_output_ln','mw_wide_plain_affine_output_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_plain_affine_output_ln').kernel('mw_wide_plain_affine_output_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),256,self.smem)))
        assert self.occupancy>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
    def __call__(self):
        self.p.floats[10].zero_();self.p.floats[11].zero_()
        self.kernel.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
