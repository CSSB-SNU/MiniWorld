"""Experimental K1 preactivation saves and B7 dW without recomputation."""
from pathlib import Path
import re
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

R = Path(__file__).resolve().parent


class SavedFront:
    def __init__(self, original, pre):
        self.__dict__.update(original.__dict__)
        a,b,slots,sk,mb=self.cfg[:5]
        assert len(self.cfg)==5 and self.normalize
        header=(F.headers()/'tmn_kernels.cuh').read_text()
        marker='  int N, Np, tiles_j, num_tiles, vec, ms_i, ms_j;'
        assert header.count(marker)==1
        header=header.replace(marker,'  __nv_bfloat16* pre;\n'+marker)
        marker='      uint32_t pk[4][2];'
        assert header.count(marker)==1
        save='''
      #pragma unroll
      for(int q=0;q<4;++q){
        int c=q*8+2*(lane&3);
        size_t ra=(size_t(b)*p.N*p.N+size_t(iA)*p.N+jA)*64+c;
        size_t rb=(size_t(b)*p.N*p.N+size_t(iB)*p.N+jB)*64+c;
        if(vA){
          *reinterpret_cast<uint32_t*>(p.pre+ra)=pack_bf16(ac[4*q],ac[4*q+1]);
          *reinterpret_cast<uint32_t*>(p.pre+ra+32)=pack_bf16(ac[4*q+16],ac[4*q+17]);
        }
        if(vB){
          *reinterpret_cast<uint32_t*>(p.pre+rb)=pack_bf16(ac[4*q+2],ac[4*q+3]);
          *reinterpret_cast<uint32_t*>(p.pre+rb+32)=pack_bf16(ac[4*q+18],ac[4*q+19]);
        }
      }
'''
        header=header.replace(marker,save+marker)
        body=header+f'''
using C=tmn::K1Cfg<256,512,false,{a},{b},{slots},{sk},-1>;
extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB)
void mw_front_saved_pre(__grid_constant__ const tmn::K1Params p){{
 tmn::sm90::k1_body<C,true,1,true,true,1>(p);
}}
'''
        flags=[f'-DMW_MINB={mb}','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1',
               '-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1','-std=c++17','-O3',
               '-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers())]
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_front_saved_pre').kernel('mw_front_saved_pre')
        self.k.set_max_dynamic_smem(self.smem)
        fields=original.params.fields.copy();fields.insert(9,pre)
        self.params=T._launch_module().Struct(fields)

    def __call__(self):
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)


class SavedSource:
    def __init__(self,plan,pre):
        original=plan.b7
        self.__dict__.update(original.__dict__)
        p=plan.p
        body=original.source_text
        assert 'compute_source' in body
        body=body.replace('CUtensorMap xn,w,dl,dr,gmap[4];','CUtensorMap xn,w,dl,dr,gmap[4],premap;')
        body=body.replace('INPUT=36864,WEIGHT=73728,DERIV=106496','INPUT=45056,WEIGHT=90112,DERIV=90112')
        body=body.replace('114688','98304')
        marker=' tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n tma_load_2d(sm+slot*INPUT+CH+4096,&p.premap,bar+slot,0,rank*p.M+row);')
        begin=body.index('  float pre[32]={};')
        end=body.index('  int ra=warp*16+lane/4;',begin)
        body=body[:begin]+'''
  float pre[32];
  #pragma unroll
  for(int j=0;j<32;++j){
    int r=warp*16+lane/4+8*((j/2)&1);
    int c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
    pre[j]=rd(reinterpret_cast<bf*>(xn+CH+4096),swz128(r,c*2)/2);
  }
'''+body[end:]
        body=body.replace(' mbar_wait(bar+2,0);named_bar_sync(1,128);',' named_bar_sync(1,128);')
        marker='   mbar_arrive_expect_tx(bar+2,CH);\n   for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);'
        assert body.count(marker)==1
        body=body.replace(marker,'')
        body=body.replace('mw_d256_b7_tma','mw_d256_b7_saved_pre')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.cubin),'mw_d256_b7_saved_pre').kernel('mw_d256_b7_saved_pre')
        self.smem=98432;self.source.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        fields=original.params.fields.copy()
        fields.insert(8,L.tensor_map(pre,[64,64],dims=[64,32*p.M],strides_bytes=[128],swizzle='128B',l2='128B'))
        self.params=L.Struct(fields)
        self.p=p

    def source_only(self):
        self.source.launch((32*self.splits,1,1),(self.source_threads,1,1),[self.params],self.smem)

    def __call__(self):
        self.source_only();self.wide_finish()
        return self.p.outputs


def enable(plan):
    pre=plan.p.x.new_empty((32,plan.p.M,64))
    plan.f.front=SavedFront(plan.f.front,pre)
    plan.b7=SavedSource(plan,pre)
    plan.saved=(*plan.saved,pre)
    return dict(name='saved_front_products',front_cubin=str(plan.f.front.cubin),
                source_cubin=str(plan.b7.cubin),saved_bytes=pre.numel()*2)
