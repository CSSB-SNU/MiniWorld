"""Ordered two-warp LN retaining only BF16 pairs across its serial handoff."""
from pathlib import Path
import re
import torch
from wide_ordered_pair_ln import OrderedPairLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PackedOrderedPairLN(OrderedPairLN):
    def __init__(self,p,rows=8,minblocks=4,narrow=False):
        super().__init__(p,16,3,False)
        assert p.D==512 and rows in (8,16)
        root=Path(__file__).resolve().parent;h=2*p.D
        body=self.source_text
        stats=2*rows*h*2+128+h*4
        sums=(stats+rows*8+127)//128*128
        self.smem=sums+640;self.rows=rows
        body=re.sub(r'constexpr int STATS=\d+;',f'constexpr int STATS={stats};',body)
        body=re.sub(r'constexpr int PAIR_SUMS=\d+;',f'constexpr int PAIR_SUMS={sums};',body)
        if rows==8:
            begin=body.index('template<bool INVERSE>');end=body.index('TMN_DEVI void put_tile',begin)
            helper=(root/'tile_transpose8.cuh').read_text().split('TMN_DEVI void put_tile')[0]
            body=body[:begin]+helper+body[end:]
            L=T._launch_module();fields=self.params.fields.copy()
            tm=lambda t:L.tensor_map(t,[rows,64,h//64],dims=[p.M,64,h//64],strides_bytes=[p.M*2,p.M*128],swizzle='none',l2='128B')
            fields[0]=tm(p.tri);fields[1]=tm(p.dt)
            fields[2]=L.tensor_map(p.tensors[9],[64,rows,h//64],dims=[64,p.M,h//64],strides_bytes=[h*2,128],swizzle='128B',l2='128B')
            self.params=L.Struct(fields)
        begin=body.index('  for(int r=pair;r<ROWS;r+=2){')
        end=body.index('  __syncthreads();transpose32<true>',begin)
        loop='''
  for(int r=pair;r<ROWS;r+=2){
   float mu=reinterpret_cast<const float*>(sm+STATS)[r],rs=reinterpret_cast<const float*>(sm+STATS)[ROWS+r];
   uint32_t tri[H/128],dn[H/128];float s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/128;++q){int c=lane+half*(H/2)+q*64;
    tri[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c)]))|(uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c+32)]))<<16);
    dn[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]))|(uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c+32)]))<<16);
   }
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]),dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs;gg[q]+=dy*z;bb[q]+=dy;
    if(half==0){float v=dy*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z;}
   }
   if(half==0){sums[pair*64+lane]=s0;sums[pair*64+32+lane]=s1;}
   #pragma unroll
   for(int q=0;q<H/128;++q)asm volatile("" : "+r"(tri[q]),"+r"(dn[q]) :: "memory");
   __syncthreads();
   if(half==1){
    s0=sums[pair*64+lane];s1=sums[pair*64+32+lane];
    #pragma unroll
    for(int q=0;q<H/64;++q){int c=lane+H/2+q*32;
     float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]),dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
     float z=(x-mu)*rs,v=dy*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z;
    }
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    if(lane==0){sums[128+pair*2]=s0;sums[128+pair*2+1]=s1;}
   }
   #pragma unroll
   for(int q=0;q<H/128;++q)asm volatile("" : "+r"(tri[q]),"+r"(dn[q]) :: "memory");
   __syncthreads();
   s0=sums[128+pair*2];s1=sums[128+pair*2+1];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]),dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs,centered=fmaf(dy,reinterpret_cast<float*>(sm+2*SB+128)[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,centered)*rs);
   }
   __syncthreads();
  }
'''
        if narrow:loop=loop.replace('__syncthreads();','named_bar_sync(2+pair,64);')
        body=body[:begin]+loop+body[end:]
        body=body.replace('__launch_bounds__(NT,LN_MINBLOCKS)',f'__launch_bounds__(NT,{minblocks})')
        body=body.replace('mw_ordered_pair_ln','mw_packed_ordered_pair_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_packed_ordered_pair_ln').kernel('mw_packed_ordered_pair_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
