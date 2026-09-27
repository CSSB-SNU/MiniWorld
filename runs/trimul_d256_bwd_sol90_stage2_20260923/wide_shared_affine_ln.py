"""Output LN with warp-owned affine sums in shared memory, no atomics there."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SharedAffineLN:
    def __init__(self,p,original,threads=256):
        root=Path(__file__).resolve().parent;h=2*p.D
        self.params=original.lp;self.threads=threads
        self.smem=128*h+128+2*(threads//32)*h*4
        assert self.smem<=232448
        body=(root/'ln.cu').read_text()
        body=body.replace('H=512',f'H={h}').replace('SB=65536',f'SB={128*h}')
        body=body.replace('NT=256',f'NT={threads}').replace('kc<8','kc<H/64')
        body=body.replace('kc+=2','kc+=NT/128').replace('r+=8','r+=NT/32')
        body=body.replace('[16]','[H/32]').replace('q<16','q<H/32')
        body=body.replace('__launch_bounds__(NT,2)','__launch_bounds__(NT,1)')
        body=body.replace('mw_d256_ln','mw_wide_shared_affine_ln')
        marker=' float gg[H/32]={},bb[H/32]={};int phase=0;'
        assert body.count(marker)==1
        body=body.replace(marker,''' volatile float* acc=reinterpret_cast<volatile float*>(sm+SB+128);
 for(int i=tid;i<2*(NT/32)*H;i+=NT)acc[i]=0;
 __syncthreads();int phase=0;''')
        marker='gg[q]+=dy*z[q];bb[q]+=dy;'
        assert body.count(marker)==1
        body=body.replace(marker,'acc[warp*H+c]+=dy*z[q];acc[(NT/32+warp)*H+c]+=dy;')
        marker=' for(int q=0;q<H/32;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}'
        assert body.count(marker)==1
        body=body.replace(marker,''' __syncthreads();
 for(int c=tid;c<H;c+=NT){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<NT/32;++w){g+=acc[w*H+c];b+=acc[(NT/32+w)*H+c];}
  atomicAdd(p.dg+c,g);atomicAdd(p.db+c,b);
 }''')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_shared_affine_ln').kernel('mw_wide_shared_affine_ln')
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
