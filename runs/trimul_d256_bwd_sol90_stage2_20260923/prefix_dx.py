"""One dX GEMM with output-gate terms first, preserving native K ordering."""
from pathlib import Path
import ctypes,torch,os
from quack.gemm import gemm
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from miniworld_engine.kernels.trimul_inproj.cuda.h100_gp import GP


class PrefixDx:
    def __init__(self,p,leaves,splits=32,tile=(128,128),pingpong=True):
        self.p=p;d=p.D
        self.input=p.x.new_empty((9*d,p.M));self.weights=p.x.new_empty((9*d,d))
        self.weight_inputs=(leaves[5],*p.weights)
        # Source publishes directly into the suffix; only dGate needs a copy.
        p.gp_all=self.input[d:].view(4,2*d,p.M);p.gp=list(p.gp_all.unbind())
        p.tensors[13:17]=p.gp;p.maps[6:10]=[W.tm(g) for g in p.gp]
        p.maps7=p.maps.copy();p.maps7[14:16]=[W.tm(p.dl.reshape(2*d,p.M)),W.tm(p.dr.reshape(2*d,p.M))]
        L=T._launch_module();p.params=L.Struct([*p.maps,*p.tensors,*p.floats,p.M,p.n])
        t7=p.tensors.copy();t7[22:24]=[p.dl,p.dr]
        p.params7=L.Struct([*p.maps7,*t7,*p.floats,p.M,p.n])
        p.gp_native=GP(p,W.gp_config(d),parameters_only=True)
        self.args=(self.input.t().unsqueeze(0),self.weights.t().unsqueeze(0),p.tensors[10].unsqueeze(0))
        self.cfg=dict(tile_M=tile[0],tile_N=tile[1],cluster_M=1,cluster_N=1,pingpong=pingpong)
        self.impl=os.environ.get('PREFIX_IMPL','cute')
        assert self.impl in ('cute','blas')
        self.copy_impl=os.environ.get('PREFIX_COPY','torch')
        assert self.copy_impl in ('torch','tma')
        if self.copy_impl=='tma':
            flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
            self.copy_cubin=T.compile(Path(__file__).with_name('prefix_transpose.cu'),flags)
            self.copy_kernel=T.load_unit(str(self.copy_cubin),'mw_prefix_transpose').kernel('mw_prefix_transpose')
            self.copy_kernel.set_max_dynamic_smem(8320)
            self.copy_params=L.Struct([W.tm(p.dg),W.tm(self.input[:d]),p.M,d])
            self.copy_grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*8
        body=(W.R/'widths.cu').read_text().split('extern "C" __global__')[0]
        body+='''
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_prefix_dx_reduce(__grid_constant__ const Params p){
 for(int c=blockIdx.x*THREADS+threadIdx.x;c<D;c+=gridDim.x*THREADS){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();ln_bwd<true>(p);
 for(int i=0;i<4;++i)reduce_w(p,(3+2*i)*D*D,H*D,p.t[17+i]);
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DWEIGHT_SPLITS={splits}',
               '-DFUSED_GP=0','-DWIDTH_GROUPS=2','-DWIDTH_N=64']
        self.cubin=T.compile_text(body,flags);self.reduce=T.load_unit(str(self.cubin),'mw_prefix_dx_reduce').kernel('mw_prefix_dx_reduce')
        drv=self.reduce.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.reduce.handle)),256,0)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
    def copy_prefix(self):
        if self.copy_impl=='tma':self.copy_kernel.launch((self.copy_grid,1,1),(128,1,1),[self.copy_params],8320)
        else:self.input[:self.p.D].copy_(self.p.dg.t())
    def pack_weights(self):torch.cat(self.weight_inputs,out=self.weights)
    def gemm_only(self):
        if self.impl=='blas':torch.mm(self.input.t(),self.weights,out=self.p.tensors[10])
        else:gemm(*self.args,None,None,**self.cfg)
    def matrix_product(self):
        self.copy_prefix();self.pack_weights();self.gemm_only()
    def __call__(self):
        self.matrix_product()
        self.reduce_only()
    def reduce_only(self):
        L=T._launch_module();drv=self.reduce.unit.drv;args=L._Packed([self.p.params7])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.reduce.handle)),self.grid,1,1,256,1,1,0,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
