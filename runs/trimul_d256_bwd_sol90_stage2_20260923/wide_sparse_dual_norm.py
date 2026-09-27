"""D512 forward emits its packed norm and the backward scalar norm together."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from lt_contract import LtBmm
from wide_split_dwp import SplitProjectionWeight

class SparseDualNorm:
    def __init__(self,p,original,capacity=None):
        self.__dict__.update(original.__dict__)
        assert p.D==512
        root=Path(__file__).resolve().parent;h=2*p.D;rows=16
        self.capacity=max(1,p.M*h//1024) if capacity is None else capacity
        self.patches=torch.empty(max(1,self.capacity),device=p.x.device,dtype=torch.int64)
        self.count=torch.zeros(1,device=p.x.device,dtype=torch.int32)
        self.smem=rows*h*4+128
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('CUtensorMap tri,norm;', 'CUtensorMap tri,norm;uint64_t* patches;unsigned* count;unsigned capacity;').replace('sm+SB);','sm+2*SB);')
        marker='   float s=0,mu,rs;'
        assert body.count(marker)==1
        body=body.replace(marker,'''   {
    float v[H/32],s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){v[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=v[q];}
    float mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=v[q]-mu;s+=z*z;}
    float rs=rsqrtf(sumwarp(s)/H+1e-5f);
    if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm+SB)[pos(r,c)]=__float2bfloat16_rn(fmaf((v[q]-mu)*rs,p.gamma[c],p.beta[c]));}
   }
'''+marker)
        # Only the first occurrence is our scalar statistics, the later one is packed.
        point='\n   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}'
        assert body.count(point)==1
        body=body.replace(point,'')
        marker='  __syncthreads();fence_proxy_async();__syncthreads();'
        assert body.count(marker)==1
        body=body.replace(marker,'''  __syncthreads();
  for(int i=tid;i<ROWS*H/2;i+=NT){int r=i/(H/2),c=(i%(H/2))*2;
   uint32_t packed=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));
   uint32_t scalar=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm+SB)+pos(r,c));
   if((packed&65535u)!=(scalar&65535u)){
    unsigned at=atomicAdd(p.count,1u);if(at<p.capacity)p.patches[at]=(uint64_t(scalar&65535u)<<32)|uint32_t((row+r)*H+c);
   }
   if((packed>>16)!=(scalar>>16)){
    unsigned at=atomicAdd(p.count,1u);if(at<p.capacity)p.patches[at]=(uint64_t(scalar>>16)<<32)|uint32_t((row+r)*H+c+1);
   }
  }
  fence_proxy_async();__syncthreads();''')
        body=body.replace('mw_wide_tile_norm','mw_wide_sparse_dual_norm')
        body+='''
struct Apply {const uint64_t* patches;const unsigned* count;unsigned capacity;bf* norm;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_apply_norm_patches(__grid_constant__ const Apply p){
 unsigned count=*p.count;if(count>p.capacity)return;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<count;i+=gridDim.x*256){uint64_t v=p.patches[i];reinterpret_cast<uint16_t*>(p.norm)[uint32_t(v)]=uint16_t(v>>32);}
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DWIDTH=512','-DOUTPUT_ROWS=16','-DOUTPUT_THREADS=128']
        self.cubin=T.compile_text(body,flags)
        self.norm=T.load_unit(str(self.cubin),'mw_wide_sparse_dual_norm').kernel('mw_wide_sparse_dual_norm');self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B')
        tm=lambda t:L.tensor_map(t,[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,tm(p.tensors[6]),self.patches,self.count,self.capacity,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])

        self.native_norm=self.norm
        self.norm=ResetNorm(self)
        self.apply=T.load_unit(str(self.cubin),'mw_wide_apply_norm_patches').kernel('mw_wide_apply_norm_patches')
        self.ap=L.Struct([self.patches,self.count,self.capacity,p.tensors[6]])
        # Overflow recomputes the exact scalar order on the GPU; no host decision.
        fallback=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        fallback=fallback.replace('int M;};','int M;const unsigned* count;unsigned capacity;};')
        fallback=fallback.replace('if constexpr(D<512)','if constexpr(true)')
        fallback=fallback.replace('extern __shared__ __align__(1024) uint8_t sm[];', 'if(*p.count<=p.capacity)return;\n extern __shared__ __align__(1024) uint8_t sm[];')
        fallback=fallback.replace('mw_wide_tile_norm','mw_wide_scalar_norm_fallback')
        self.fallback_cubin=T.compile_text(fallback,flags)
        self.fallback=T.load_unit(str(self.fallback_cubin),'mw_wide_scalar_norm_fallback').kernel('mw_wide_scalar_norm_fallback')
        self.fallback_smem=rows*h*2+128;self.fallback.set_max_dynamic_smem(self.fallback_smem)
        self.fp=L.Struct([tri,tm(p.tensors[6]),p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M,self.count,self.capacity])

class ResetNorm:
    def __init__(self,owner):self.owner=owner
    def launch(self,*args):
        self.owner.count.zero_();self.owner.native_norm.launch(*args)

class PatchBackwardNorm:
    def __init__(self,owner):self.owner=owner
    def launch(self,*args):
        o=self.owner
        o.apply.launch((o.grid,1,1),(256,1,1),[o.ap],0)
        o.fallback.launch((o.grid,1,1),(128,1,1),[o.fp],o.fallback_smem)
