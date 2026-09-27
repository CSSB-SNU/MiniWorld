"""Bound scalar LN temporaries inside exact PTX steps, retaining BF16 pairs."""
import torch
from wide_packed_retained_ln import PackedRetainedLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PtxRetainedLN(PackedRetainedLN):
    def __init__(self,p,original,rows=16,minblocks=3,output_ptx=True):
        super().__init__(p,original,rows,minblocks)
        body=self.source_text
        start=body.index('   #pragma unroll\n   for(int q=0;q<H/32;++q)')
        end=body.index('   s0=sumwarp(s0)/H;',start)
        gamma=('reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]' if self.kind=='CachedLN' else 'reinterpret_cast<float*>(storage+4*SB+128)[c]')
        body=body[:start]+'''   static_for<H/32>([&](auto qq){constexpr int q=decltype(qq)::value;int c=lane+q*32;
    uint32_t xb=(q&1)?(tri[q/2]&0xffff0000u):(tri[q/2]<<16);
    uint32_t yb=(q&1)?(dn[q/2]&0xffff0000u):(dn[q/2]<<16);
    asm volatile("{.reg .f32 x,y,z,v;mov.b32 x,%4;mov.b32 y,%5;"
                 "sub.rn.f32 z,x,%6;mul.rn.f32 z,z,%7;mul.rn.f32 v,y,%8;"
                 "add.rn.f32 %0,%0,v;fma.rn.f32 %1,v,z,%1;"
                 "fma.rn.f32 %2,y,z,%2;add.rn.f32 %3,%3,y;}"
        :"+f"(s0),"+f"(s1),"+f"(gg[q]),"+f"(bb[q])
        :"r"(xb),"r"(yb),"f"(mu),"f"(rs),"f"(GAMMA):"memory");
   });
'''.replace('GAMMA',gamma)+body[end:]
        if output_ptx:
            start=body.index('   #pragma unroll\n   for(int q=0;q<H/32;++q)')
            end=body.index('  __syncthreads();transpose32<true>',start)
            body=body[:start]+'''   static_for<H/32>([&](auto qq){constexpr int q=decltype(qq)::value;int c=lane+q*32;
    uint32_t xb=(q&1)?(tri[q/2]&0xffff0000u):(tri[q/2]<<16);
    uint32_t yb=(q&1)?(dn[q/2]&0xffff0000u):(dn[q/2]<<16);unsigned short out;
    asm volatile("{.reg .f32 x,y,z,v,a;mov.b32 x,%1;mov.b32 y,%2;"
                 "sub.rn.f32 z,x,%3;mul.rn.f32 z,z,%4;neg.f32 a,%6;"
                 "fma.rn.f32 v,y,%5,a;neg.f32 z,z;fma.rn.f32 v,z,%7,v;"
                 "mul.rn.f32 v,v,%4;cvt.rn.bf16.f32 %0,v;}"
        :"=h"(out):"r"(xb),"r"(yb),"f"(mu),"f"(rs),"f"(GAMMA),"f"(s0),"f"(s1):"memory");
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__ushort_as_bfloat16(out);
   });
  }
'''.replace('GAMMA',gamma)+body[end:]
        body=body.replace('mw_packed_retained_ln','mw_ptx_retained_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_ptx_retained_ln').kernel('mw_ptx_retained_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
