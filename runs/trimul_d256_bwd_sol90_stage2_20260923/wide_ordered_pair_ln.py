"""Two contiguous channel owners pass the exact serial row sums between warps."""
from pathlib import Path
import torch
from wide_cached_ln import CachedLN
from wide_permuted_stats_ln import PermutedStatsLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class OrderedPairLN(PermutedStatsLN):
    def __init__(self,p,rows=16,minblocks=3,narrow=False):
        base=CachedLN(p,rows,True,False,2)
        super().__init__(p,base,rows,2)
        body=self.source_text
        marker=' int tid=threadIdx.x,lane=tid%32,warp=tid/32;'
        where=body.index(marker,body.index('void mw_permuted_stats_ln('))
        where+=len(marker)
        body=body[:where]+'\n int pair=warp/2,half=warp%2;float* sums=reinterpret_cast<float*>(sm+PAIR_SUMS);'+body[where:]
        body=body.replace('float gg[H/32]={},bb[H/32]={};','float gg[H/64]={},bb[H/64]={};')
        start=body.index('  for(int r=warp;r<ROWS;r+=4){')
        end=body.index('  __syncthreads();transpose32<true>',start)
        body=body[:start]+'''
  for(int r=pair;r<ROWS;r+=2){
   float mu=reinterpret_cast<const float*>(sm+STATS)[r],rs=reinterpret_cast<const float*>(sm+STATS)[ROWS+r];
   float z[H/64],dy[H/64],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    z[q]=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    dy[q]=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
    gg[q]+=dy[q]*z[q];bb[q]+=dy[q];
    if(half==0){float v=dy[q]*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z[q];}
   }
   if(half==0){sums[pair*64+lane]=s0;sums[pair*64+32+lane]=s1;}
   __syncthreads();
   if(half==1){
    s0=sums[pair*64+lane];s1=sums[pair*64+32+lane];
    #pragma unroll
    for(int q=0;q<H/64;++q){int c=lane+H/2+q*32;float v=dy[q]*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z[q];}
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    if(lane==0){sums[pair*64]=s0;sums[pair*64+1]=s1;}
   }
   __syncthreads();
   s0=sums[pair*64];s1=sums[pair*64+1];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    float centered=fmaf(dy[q],reinterpret_cast<float*>(sm+2*SB+128)[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);
   }
   __syncthreads();
  }
'''+body[end:]
        old=' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);'
        assert body.count(old)==1
        body=body.replace(old,'''
 float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;acc[pair*H+c]=gg[q];acc[(2+pair)*H+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<H;c+=NT){atomicAdd(p.dg+c,acc[c]+acc[H+c]);atomicAdd(p.db+c,acc[2*H+c]+acc[3*H+c]);}
''')
        sum_offset=(self.smem+127)//128*128
        self.smem=sum_offset+512
        body=body.replace('constexpr int H=2*WIDTH',f'constexpr int PAIR_SUMS={sum_offset};\nconstexpr int H=2*WIDTH')
        if narrow:
            start=body.index('  for(int r=pair;r<ROWS;r+=2){')
            end=body.index('  __syncthreads();transpose32<true>',start)
            body=body[:start]+body[start:end].replace('__syncthreads();','named_bar_sync(2+pair,64);')+body[end:]
        body=body.replace('mw_permuted_stats_ln','mw_ordered_pair_ln')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_ordered_pair_ln').kernel('mw_ordered_pair_ln')
        self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
