"""Input LN/weight reduction with CTA aggregation of affine gradients."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W


class InputReduce:
    def __init__(self,p,threads=256,minblocks=2):
        self.p=p;self.threads=threads;self.smem=2*(threads//32)*p.D*4
        root=Path(__file__).resolve().parent
        body=(W.R/'widths.cu').read_text().split('extern "C" __global__')[0]
        marker='template<bool INPUT> TMN_DEVI void ln_bwd(const Params& p){'
        assert body.count(marker)==1
        body=body.replace(marker,(root/'ln_aggregate.cuh').read_text()+'\ntemplate<bool INPUT> TMN_DEVI void ln_bwd(const Params& p,uint8_t* sm){')
        marker=' for(int c=lane;c<C;c+=32){atomicAdd(p.f[INPUT?8:10]+c,gg[c/32]);atomicAdd(p.f[INPUT?9:11]+c,bb[c/32]);}'
        assert body.count(marker)==1
        body=body.replace(marker,' aggregate_ln<C,THREADS>(gg,bb,reinterpret_cast<float*>(sm),p.f[INPUT?8:10],p.f[INPUT?9:11]);')
        body+=f'''
extern "C" __global__ __launch_bounds__(THREADS,{minblocks})
void mw_wide_input_ln_reduce(__grid_constant__ const Params p){{
 extern __shared__ __align__(128) uint8_t sm[];
 for(int c=blockIdx.x*THREADS+threadIdx.x;c<D;c+=gridDim.x*THREADS){{p.f[8][c]=0;p.f[9][c]=0;}}
 __syncthreads();cooperative_groups::this_grid().sync();ln_bwd<true>(p,sm);
 for(int i=0;i<4;++i)reduce_w(p,(3+2*i)*D*D,H*D,p.t[17+i]);
}}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DWEIGHT_SPLITS=32',
               '-DFUSED_GP=0',f'-DWIDTH_GROUPS={threads//128}','-DWIDTH_N=64']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_input_ln_reduce').kernel('mw_wide_input_ln_reduce')
        self.k.set_max_dynamic_smem(self.smem);drv=self.k.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.p.params7])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
