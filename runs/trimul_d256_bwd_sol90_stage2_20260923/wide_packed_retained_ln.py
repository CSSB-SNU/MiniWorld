"""Keep exact BF16 triangle/dNorm pairs through the two scalar LN passes."""
from pathlib import Path
import torch
from wide_permuted_stats_ln import PermutedStatsLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PackedRetainedLN(PermutedStatsLN):
    def __init__(self,p,original,rows=16,minblocks=3):
        super().__init__(p,original,rows,minblocks)
        body=self.source_text
        begin=body.index('  for(int r=warp;r<ROWS;r+=4){')
        end=body.index('  __syncthreads();transpose32<true>',begin)
        old=body[begin:end]
        init=old[old.index('   float mu='):old.index('   #pragma unroll')]
        gamma=('reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]' if self.kind=='CachedLN' else 'reinterpret_cast<float*>(storage+4*SB+128)[c]')
        if self.kind!='CachedLN':
            # PrefetchLN uses the immutable affine array after its double tiles.
            assert gamma in old,old
        new='''  for(int r=warp;r<ROWS;r+=4){
'''+init+'''   uint32_t tri[H/64],dn[H/64];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+q*64;
    tri[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c)]))|
           (uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c+32)]))<<16);
    dn[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]))|
          (uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c+32)]))<<16);
   }
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]);
    float dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs,v=dy*GAMMA;
    s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   // Opaque, value-preserving boundaries prevent FP32 expansion from
   // surviving across the two passes. Arithmetic and channel order match.
   #pragma unroll
   for(int q=0;q<H/64;++q)asm volatile("" : "+r"(tri[q]),"+r"(dn[q]) :: "memory");
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]);
    float dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs,centered=fmaf(dy,GAMMA,-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,centered)*rs);
   }
  }
'''
        new=new.replace('GAMMA',gamma)
        body=body[:begin]+new+body[end:]
        body=body.replace('mw_permuted_stats_ln','mw_packed_retained_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_packed_retained_ln').kernel('mw_packed_retained_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
