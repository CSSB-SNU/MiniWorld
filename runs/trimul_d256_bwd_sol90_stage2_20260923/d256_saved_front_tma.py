"""Save BF16 projection fragments through the existing shared staging allocation."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class D256SavedFrontTma:
    def __init__(self,original,pre):
        self.__dict__.update(original.__dict__)
        d=self.x.shape[-1];a,b,slots,sk,mb=self.cfg[:5]
        assert d==256 and original.params.fields[7] is not None
        assert b==64
        shared=self.cfg[5] if len(self.cfg)>5 else 0;stream=int(shared==2)
        head=(F.headers()/'tmn_kernels.cuh').read_text()
        marker='  int N, Np, tiles_j, num_tiles, vec, ms_i, ms_j;'
        assert head.count(marker)==1
        head=head.replace(marker,'  CUtensorMap premap;\n'+marker)
        source=(F.R/'front_shared.cuh').read_text().replace('#include "tmn_kernels.cuh"','') if shared else head
        marker='      uint32_t pk[4][2];';assert source.count(marker)==1
        source=source.replace(marker,'''
      static_assert(TMAST,"TMA store completion must be drained at kernel exit");
      // Reuse both 4KB output staging slots for one 8KB preactivation tile.
      if(st_elect)tma_store_wait_read<0>();
      named_bar_sync(bar_id,128);
      uint8_t* savebuf=sStage+cw*8192;
      #pragma unroll
      for(int q=0;q<4;++q){int c=q*8+2*(lane&3),r=16*wiw+(lane>>2);
       *reinterpret_cast<uint32_t*>(savebuf+swz128(r,c*2))=pack_bf16(ac[4*q],ac[4*q+1]);
       *reinterpret_cast<uint32_t*>(savebuf+swz128(r+8,c*2))=pack_bf16(ac[4*q+2],ac[4*q+3]);
       *reinterpret_cast<uint32_t*>(savebuf+swz128(r,(c+32)*2))=pack_bf16(ac[4*q+16],ac[4*q+17]);
       *reinterpret_cast<uint32_t*>(savebuf+swz128(r+8,(c+32)*2))=pack_bf16(ac[4*q+18],ac[4*q+19]);
      }
      fence_proxy_async();named_bar_sync(bar_id,128);
      if(st_elect){
       int savedrow=b*p.N*p.N+iw*p.N+jw;
       asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.premap),"r"(smem_u32(savebuf)),"r"(0),"r"(savedrow):"memory");
       tma_store_commit();
      }
'''+marker)
        marker='      const uint32_t sbuf = stage_u';assert source.count(marker)==1
        source=source.replace(marker,'      if(st_elect)tma_store_wait_read<0>();\n      named_bar_sync(bar_id,128);\n'+marker)
        if shared:source=head+'\n'+source
        function='mw_k1_shared' if shared else 'k1_body'
        source+=f'''
using C=tmn::K1Cfg<{d},{2*d},false,{a},{b},{slots},{sk},{1 if stream else -1}>;
extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB)
void mw_d256_front_pre_tma_stats(__grid_constant__ const tmn::K1Params p){{tmn::sm90::{function}<C,true,{int(self.normalize)},true,{str(self.normalize).lower()},1>(p);}}
'''
        flags=[f'-DMW_MINB={mb}',f'-DMW_K1_STREAM={stream}','-DTMN_SIGMOID_TANH=1','-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1',
               '-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-I'+str(F.R)]
        self.cubin=T.compile_text(source,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_front_pre_tma_stats').kernel('mw_d256_front_pre_tma_stats');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();prem=L.tensor_map(pre,[64,64],dims=[64,pre.numel()//64],strides_bytes=[128],swizzle='128B',l2='128B')
        fields=original.params.fields.copy();fields.insert(9,prem);self.params=L.Struct(fields)
    def __call__(self):
        if not self.normalize:F.normalize_into(self.xn,self.x,self.gi,self.bi)
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
