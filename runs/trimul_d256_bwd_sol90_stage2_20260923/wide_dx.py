"""Explicit wider dX tile for the new wide B7; preserves K accumulation order."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W


class WideDx:
    def __init__(self,p,n=128,groups=2,splits=32):
        self.p=p;self.threads=128*groups;self.smem=2*8192*(1+groups*(n//64))+128
        assert n in (64,128) and groups in (1,2)
        body=(W.R/'widths.cu').read_text().split('extern "C" __global__')[0]
        body+='''
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide_dx_finish(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+2*STAGE);
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}
 for(int c=blockIdx.x*THREADS+threadIdx.x;c<D;c+=gridDim.x*THREADS){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();grid.sync();
 front_dx(p,sm,bar,phase);__threadfence();grid.sync();ln_bwd<true>(p);
 for(int i=0;i<4;++i)reduce_w(p,(3+2*i)*D*D,H*D,p.t[17+i]);
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DWEIGHT_SPLITS={splits}',
               '-DFUSED_GP=0',f'-DWIDTH_GROUPS={groups}',f'-DWIDTH_N={n}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_dx_finish').kernel('mw_wide_dx_finish');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.p.params7])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
