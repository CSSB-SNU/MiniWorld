"""D384/512 pilot of our D256 dense GEMM and TMA output-LN B1 schedule."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

R=Path(__file__).resolve().parent


class ExactProjectionWeight:
    """Retain the preceding 32-way FP32 accumulation for long D512 dWproj."""
    def __init__(self,p):
        self.p=p
        body=(T.SOURCES/'wide_base/widths.cu').read_text().split('template<bool INPUT>')[0]
        body+='''
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide512_dwp(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+2*STAGE);
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}
 __syncthreads();int phase=0;
 weight(p,4,3,D,H,0,sm,bar,phase);
}
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide512_dwp_reduce(__grid_constant__ const Params p){reduce_w(p,0,D*H,p.t[21]);}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),'-DWIDTH=512','-DWEIGHT_SPLITS=32']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide512_dwp')
        self.k=unit.kernel('mw_wide512_dwp');self.k.set_max_dynamic_smem(49280)
        self.reduce=unit.kernel('mw_wide512_dwp_reduce')
    def __call__(self):
        self.k.launch((528,1,1),(256,1,1),[self.p.params],49280)
        self.reduce.launch((264,1,1),(256,1,1),[self.p.params],0)


class WideB1:
    def __init__(self,p,leaves):
        self.p=p;d=p.D;h=2*d;self.wp=leaves[6];self.wg=leaves[5]
        self.use_saved_norm=False
        self.proj=torch.empty_like(p.dg);self.gate=torch.empty_like(p.dg)
        self.smem=128*h+128;self.threads=128 if d==384 else 256
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0']
        if d==384:flags.append('-DTMN_SIGMOID_TANH=1')
        # D384's preceding streamed B1 uses tanh sigmoid; the selected D512
        # legacy B1 uses ex2/rcp. Preserve each control's derivative arithmetic.
        def widen(body):
            body=body.replace('H=512',f'H={h}').replace('SB=65536',f'SB={128*h}')
            body=body.replace('NT=256',f'NT={self.threads}').replace('kc<8','kc<H/64')
            body=body.replace('kc+=2','kc+=NT/128').replace('r+=8','r+=NT/32')
            body=body.replace('[16]','[H/32]').replace('q<16','q<H/32')
            body=body.replace('__launch_bounds__(NT,2)','__launch_bounds__(NT,1)')
            return body.replace('mw_d256_',f'mw_wide{d}_')
        body=widen((R/'norm.cu').read_text())
        self.norm_cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.norm_cubin),f'mw_wide{d}_norm')
        self.norm=unit.kernel(f'mw_wide{d}_norm');self.norm.set_max_dynamic_smem(self.smem)
        self.epi=unit.kernel(f'mw_wide{d}_gate_grad')
        body=widen((R/'ln.cu').read_text())
        body=body.replace('extern "C" __global__',(R/'ln_aggregate.cuh').read_text()+'\nextern "C" __global__')
        marker=' for(int q=0;q<H/32;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}'
        assert body.count(marker)==1
        body=body.replace(marker,' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);')
        self.ln_cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.ln_cubin),f'mw_wide{d}_ln')
        self.ln=unit.kernel(f'mw_wide{d}_ln');self.ln.set_max_dynamic_smem(self.smem)
        L=T._launch_module();tm=lambda t:F.tm(t,[64,64],[p.M,h],[p.M*2])
        self.np=L.Struct([tm(p.tri),p.maps[3],p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])
        self.ep=L.Struct([self.proj,self.gate,p.dy,p.ds,p.tensors[7],p.dg,p.M*d,p.n*d])
        self.lp=L.Struct([tm(p.tri),tm(p.dt),p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
        drv=self.ln.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.ln.handle)),self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        self.exact_dwp=ExactProjectionWeight(p) if d==512 and p.n==768 else None

    def __call__(self):
        p=self.p
        if not self.use_saved_norm:self.norm.launch((self.grid,1,1),(self.threads,1,1),[self.np],self.smem)
        torch.mm(p.tensors[6],self.wp.t(),out=self.proj)
        torch.mm(p.xn.reshape(p.M,p.D),self.wg.t(),out=self.gate)
        self.epi.launch((1056,1,1),(256,1,1),[self.ep],0)
        torch.mm(p.tensors[7],self.wp,out=p.tensors[9])
        if self.exact_dwp is None:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
        else:self.exact_dwp()
        torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg)
        L=T._launch_module();drv=self.ln.unit.drv;args=L._Packed([self.lp])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.ln.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
