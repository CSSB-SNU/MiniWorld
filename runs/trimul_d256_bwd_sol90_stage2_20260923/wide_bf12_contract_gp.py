"""Decode lossless packed preactivations at contraction's existing GP consumer."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PackedPre:
    def __init__(self,pre):
        self.pre=pre;d8,m=pre.shape
        self.mant=torch.empty((d8,m),device=pre.device,dtype=torch.uint8)
        self.exps=torch.empty((d8,m//2),device=pre.device,dtype=torch.uint8)
        self.escapes=torch.zeros((),device=pre.device,dtype=torch.int64)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v']
        self.cubin=T.compile(Path(__file__).with_name('pack_pre_bf12.cu'),flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_pack_pre_bf12').kernel('mw_pack_pre_bf12')
        self.params=T._launch_module().Struct([pre,self.mant,self.exps,self.escapes,pre.numel()//2])
    def __call__(self):
        self.escapes.zero_();self.kernel.launch((1056,1,1),(256,1,1),[self.params],0)

class PackedContractGP:
    def __init__(self,plan,packed):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__);self.packed=packed;p=plan.p;n=p.n
        body=prior.source_text
        start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
        body=body[:start]+' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'+body[end:]
        marker='CUtensorMap premap,maskmap,outmap[4];int N;'
        assert body.count(marker)==1
        body=body.replace(marker,'CUtensorMap premap,maskmap,outmap[4],mantmap,expmap;int N;')
        helper='''
TMN_DEVI uint32_t packed_sw32(uint32_t v){return v^(((v>>7)&1u)<<4);}
TMN_DEVI uint32_t decode_pair(unsigned mant,unsigned codes,const bf* original){
 unsigned ca=codes&15u,cb=codes>>4;
 unsigned a=(mant&127u)|((mant&128u)<<8)|((ca?ca+115u:0u)<<7);
 unsigned b=((mant>>8)&127u)|(mant&32768u)|((cb?cb+115u:0u)<<7);
 if(ca==15u)a=__bfloat16_as_ushort(original[0]);
 if(cb==15u)b=__bfloat16_as_ushort(original[1]);
 return a|(b<<16);
}
'''
        body=body.replace('template<int MODE> TMN_DEVI void load_input',helper+'\ntemplate<int MODE> TMN_DEVI void load_input')
        body=body.replace('mbar_arrive_expect_tx(bar+6,98304);','mbar_arrive_expect_tx(bar+6,81920);')
        begin=body.index('   tma_load_2d(sm+q,&p.premap');end=body.index('\n  }',begin)
        body=body[:begin]+'''
   int t=wm*2+wn,pr=(rank*64+pc)*p.N+mi+64*wm;
   tma_load_2d(sm+t*4096,&p.mantmap,bar+6,ni+64*wn,pr);
   tma_load_2d(sm+16384+t*2048,&p.expmap,bar+6,(ni+64*wn)/2,pr);
   tma_load_2d(sm+24576+t*4096,&p.mantmap,bar+6,ni+64*wn,pr+32*p.N);
   tma_load_2d(sm+40960+t*2048,&p.expmap,bar+6,(ni+64*wn)/2,pr+32*p.N);
   tma_load_2d(sm+49152+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);'''+body[end:]
        body=body.replace('sm+65536+off','sm+49152+off')
        marker='  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+32768+off);'
        assert body.count(marker)==1
        body=body.replace(marker,'''
  int t=WG*2+c/64;unsigned mo=sw64(r*64+c%64),eo=packed_sw32(r*32+(c%64)/2);
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
  int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
  size_t original=(size_t(rank*64+pc)*p.N+mi+64*WG+r)*p.N+ni+c;
  unsigned gm=*reinterpret_cast<uint16_t*>(sm+t*4096+mo),ge=sm[16384+t*2048+eo];
  unsigned pm=*reinterpret_cast<uint16_t*>(sm+24576+t*4096+mo),pe=sm[40960+t*2048+eo];
  uint32_t gr=decode_pair(gm,ge,p.pre+original),pr=decode_pair(pm,pe,p.pre+original+size_t(32)*p.N*p.N);
''')
        marker='  *reinterpret_cast<uint32_t*>(sm+off)=dp;*reinterpret_cast<uint32_t*>(sm+32768+off)=dg;'
        assert body.count(marker)==1
        body=body.replace(marker,'  v[j]=__uint_as_float(dp);v[j+1]=__uint_as_float(dg);')
        marker=' });\n}\ntemplate<int MODE> TMN_DEVI void run'
        assert body.count(marker)==1
        body=body.replace(marker,'''
 });
 __syncthreads();
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(WG*2+c/64)*8192+swz128(r,(c%64)*2);
  *reinterpret_cast<uint32_t*>(sm+off)=__float_as_uint(v[j]);
  *reinterpret_cast<uint32_t*>(sm+32768+off)=__float_as_uint(v[j+1]);
 });
}
template<int MODE> TMN_DEVI void run''')
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_bf12_contract_gp')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_bf12_contract_gp').kernel('mw_wide_bf12_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        mant=L.tensor_map(packed.mant,[64,64],dims=[n,8*p.D*n],strides_bytes=[n],swizzle='64B',l2='128B')
        exps=L.tensor_map(packed.exps,[32,64],dims=[n//2,8*p.D*n],strides_bytes=[n//2],swizzle='32B',l2='128B')
        fields=self.params.fields.copy();fields[-1:-1]=[mant,exps];self.params=L.Struct(fields)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
