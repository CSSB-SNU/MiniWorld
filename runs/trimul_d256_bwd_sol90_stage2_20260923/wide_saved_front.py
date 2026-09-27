"""Wide-only pilot: forward preactivation saves and exact split input dW."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from lt_contract import LtBmm


class SavedFront:
    def __init__(self,original,pre):
        self.__dict__.update(original.__dict__)
        d=self.x.shape[-1];a,b,slots,sk,mb=self.cfg[:5]
        shared=self.cfg[5] if len(self.cfg)>5 else 0;stream=int(shared==2)
        head=(F.headers()/'tmn_kernels.cuh').read_text()
        marker='  int N, Np, tiles_j, num_tiles, vec, ms_i, ms_j;'
        assert head.count(marker)==1
        head=head.replace(marker,'  __nv_bfloat16* pre;\n'+marker)
        source=(F.R/'front_shared.cuh').read_text().replace('#include "tmn_kernels.cuh"','') if shared else head
        marker='      uint32_t pk[4][2];';assert source.count(marker)==1
        source=source.replace(marker,'''
      #pragma unroll
      for(int q=0;q<4;++q){int c=q*8+2*(lane&3);
        size_t ra=(size_t(b)*p.N*p.N+size_t(iA)*p.N+jA)*64+c;
        size_t rb=(size_t(b)*p.N*p.N+size_t(iB)*p.N+jB)*64+c;
        if(vA){*reinterpret_cast<uint32_t*>(p.pre+ra)=pack_bf16(ac[4*q],ac[4*q+1]);*reinterpret_cast<uint32_t*>(p.pre+ra+32)=pack_bf16(ac[4*q+16],ac[4*q+17]);}
        if(vB){*reinterpret_cast<uint32_t*>(p.pre+rb)=pack_bf16(ac[4*q+2],ac[4*q+3]);*reinterpret_cast<uint32_t*>(p.pre+rb+32)=pack_bf16(ac[4*q+18],ac[4*q+19]);}
      }
'''+marker)
        if shared:source=head+'\n'+source
        function='mw_k1_shared' if shared else 'k1_body'
        source+=f'''
using C=tmn::K1Cfg<{d},{2*d},false,{a},{b},{slots},{sk},{1 if stream else -1}>;
extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB)
void mw_wide_front_pre(__grid_constant__ const tmn::K1Params p){{tmn::sm90::{function}<C,true,{int(self.normalize)},false,{str(self.normalize).lower()},1>(p);}}
'''
        flags=[f'-DMW_MINB={mb}',f'-DMW_K1_STREAM={stream}','-DTMN_SIGMOID_TANH=1','-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1',
               '-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-I'+str(F.R)]
        self.cubin=T.compile_text(source,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_front_pre').kernel('mw_wide_front_pre');self.k.set_max_dynamic_smem(self.smem)
        fields=original.params.fields.copy();fields.insert(9,pre);self.params=T._launch_module().Struct(fields)

    def __call__(self):
        if not self.normalize:F.normalize_into(self.xn,self.x,self.gi,self.bi)
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)


class SavedSource:
    def __init__(self,p,pre,mask):
        root=Path(__file__).resolve().parent;d=p.D;self.p=p
        body=(root/'wide_saved_front_gp.cu').read_text().replace('// PACKED_GLU',(root/'packed_glu.cuh').read_text().replace('s+32768','s+8192'))
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_saved_front_gp').kernel('mw_wide_saved_front_gp')
        self.smem=20480+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();dy=lambda t:L.tensor_map(t,[64,32],dims=[p.M,2*d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        prem=L.tensor_map(pre,[64,64],dims=[64,(d//8)*p.M],strides_bytes=[128],swizzle='128B',l2='128B')
        self.params=L.Struct([prem,dy(p.dl),dy(p.dr),*[dy(t) for t in p.gp],mask,p.M])
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*4
        step=p.M//32
        self.partial=p.floats[7].reshape(-1)[3*d*d:].as_strided((32,8*d,d),(11*d*d,d,1))
        a=p.gp_all.as_strided((32,8*d,step),(step,p.M,1));b=p.xn.as_strided((32,step,d),(step*d,d,1))
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.matmul=LtBmm(a,b,self.partial,self.workspace)

    def derivatives(self):self.k.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)
    def __call__(self):self.derivatives();self.matmul()
