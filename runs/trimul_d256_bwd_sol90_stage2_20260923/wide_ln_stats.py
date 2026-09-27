"""Output LN row statistics and affine gradients; defer dTriangle publication."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class LNStats:
    def __init__(self,p,dn,rows):
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=128
        self.smem=rows*h*4+128+h*4
        self.s0=torch.empty(p.M,device=p.x.device,dtype=torch.float32);self.s1=torch.empty_like(self.s0)
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        body=(root/'wide_tile_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers).replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        body=body.replace('int M;};','int M;float *s0,*s1;};')
        body=body.replace('&p.dnmap,bar,c,row','&p.dnmap,bar,row,c')
        body=body.replace('transpose32<false>(sm);','transpose32<false>(sm);transpose32<false>(sm+SB);')
        start=body.index('   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;')
        end=body.index('\n aggregate_ln',start)
        body=body[:start]+'''   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   if(lane==0){p.s0[row+r]=s0;p.s1[row+r]=s1;}
  }
  __syncthreads();
 }
'''+body[end:]
        body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+2*SB+128)[c]')
        marker=' __syncthreads();cooperative_groups::this_grid().sync();'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+2*SB+128)[c]=p.gamma[c];\n'+marker)
        body=body.replace('mw_wide_tile_ln','mw_wide_ln_stats')
        body+='''
struct Reconstruct{const bf *tri,*dn;const float *mu,*rs,*gamma,*s0,*s1;bf* dt;int M;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_reconstruct_dt(__grid_constant__ const Reconstruct p){
 for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<size_t(H)*p.M/2;i+=size_t(gridDim.x)*256){
  int c=i/(p.M/2),row=(i%(p.M/2))*2;
  uint32_t raw=reinterpret_cast<const uint32_t*>(p.tri)[i],dy=reinterpret_cast<const uint32_t*>(p.dn)[i];
  float za=(bf16lo(raw)-p.mu[row])*p.rs[row],zb=(bf16hi(raw)-p.mu[row+1])*p.rs[row+1];
  float a=fmaf(-za,p.s1[row],fmaf(bf16lo(dy),p.gamma[c],-p.s0[row]))*p.rs[row];
  float b=fmaf(-zb,p.s1[row+1],fmaf(bf16hi(dy),p.gamma[c],-p.s0[row+1]))*p.rs[row+1];
  reinterpret_cast<uint32_t*>(p.dt)[i]=pack_bf16(a,b);
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0','-DLN_MINBLOCKS=2']
        self.cubin=T.compile_text(body,flags)
        unit=T.load_unit(str(self.cubin),'mw_wide_ln_stats');self.kernel=unit.kernel('mw_wide_ln_stats');self.kernel.set_max_dynamic_smem(self.smem)
        self.reconstruct=unit.kernel('mw_wide_reconstruct_dt')
        drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='64B' if rows==32 else '32B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),tm(dn),dn,p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M,self.s0,self.s1])
        self.rp=L.Struct([p.tri,dn,p.floats[5],p.floats[6],p.floats[2],self.s0,self.s1,p.dt,p.M])
    def statistics(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,128,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
    def __call__(self):
        self.statistics();self.reconstruct.launch((self.grid,1,1),(256,1,1),[self.rp],0)
