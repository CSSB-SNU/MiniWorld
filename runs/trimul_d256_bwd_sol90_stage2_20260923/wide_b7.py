"""D384/512 explicit fused derivative/dW source plus native dX/LN finish."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'


class WideB7:
    def __init__(self,p,mask,splits=32):
        d=p.D;assert d in (384,512)
        self.p=p;self.splits=splits;self.source_smem=3*128*d+16384+128
        body=(R/'wide_source.cu').read_text().replace('// MMA_HELPERS',(PRE/'mma.cuh').read_text())
        body=body.replace('// PACKED_GLU',(R/'packed_glu.cuh').read_text().replace('s+32768','s+CH'))
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DWEIGHT_SPLITS={splits}']
        self.source_cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.source_cubin),'mw_wide_b7_source').kernel('mw_wide_b7_source')
        self.source.set_max_dynamic_smem(self.source_smem)
        L=T._launch_module();dy=lambda t:L.tensor_map(t,[64,32],dims=[p.M,2*d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),*[dy(t) for t in p.gp],mask,p.floats[7],p.M])
        body=(W.R/'widths.cu').read_text().split('extern "C" __global__')[0]
        body+='''
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide_b7_finish(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+2*STAGE);
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}
 for(int c=blockIdx.x*THREADS+threadIdx.x;c<D;c+=gridDim.x*THREADS){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();grid.sync();
 front_dx(p,sm,bar,phase);__threadfence();grid.sync();ln_bwd<true>(p);
 for(int i=0;i<4;++i)reduce_w(p,(3+2*i)*D*D,H*D,p.t[17+i]);
}
'''
        self.finish_cubin=T.compile_text(body,flags+['-DFUSED_GP=0','-DWIDTH_GROUPS=2','-DWIDTH_N=64'])
        self.finish=T.load_unit(str(self.finish_cubin),'mw_wide_b7_finish').kernel('mw_wide_b7_finish')
        self.finish_smem=49280;self.finish.set_max_dynamic_smem(self.finish_smem)
        drv=self.finish.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)),256,self.finish_smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ

    def source_only(self):self.source.launch((self.p.D//8*self.splits,1,1),(256,1,1),[self.params],self.source_smem)
    def finish_only(self):W.launch(self.finish,self.p.params7,self.grid,D=self.p.D)
    def __call__(self):
        self.source_only();self.finish_only();return self.p.outputs
