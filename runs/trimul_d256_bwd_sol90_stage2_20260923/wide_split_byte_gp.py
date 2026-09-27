"""Lossless BF16 byte planes; low-entropy exponent storage opts into hardware ILC."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from inline_compressed_pool import InlineCompressedPool

class SplitBytePre:
    def __init__(self,pre,xor=127,interleaved=False):
        self.pre=pre;self.xor=xor
        self.pool=InlineCompressedPool()
        if interleaved:
            with self.pool.context():self.storage=torch.empty((2,*pre.shape),device=pre.device,dtype=torch.uint8)
            self.low,self.high=self.storage.unbind()
        else:
            self.low=torch.empty_like(pre,dtype=torch.uint8)
            with self.pool.context():self.high=torch.empty_like(pre,dtype=torch.uint8)
        body='''
#include <stdint.h>
struct Params{const uint32_t* src;uint16_t* low;uint16_t* high;int pairs;};
extern "C" __global__ void mw_pack_split_byte(__grid_constant__ const Params p){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<p.pairs;i+=gridDim.x*blockDim.x){
  uint32_t v=p.src[i],a=v&65535u,b=v>>16;
  p.low[i]=uint16_t((a&127u)|((a>>8)&128u)|((b&127u)<<8)|(b&32768u));
  p.high[i]=uint16_t((((a>>7)&255u)^XOR)|((((b>>7)&255u)^XOR)<<8));
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',f'-DXOR={xor}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_pack_split_byte').kernel('mw_pack_split_byte')
        self.params=T._launch_module().Struct([pre,self.low,self.high,pre.numel()//2])
    def __call__(self):self.k.launch((1056,1,1),(256,1,1),[self.params],0)

class SplitByteGP:
    def __init__(self,plan,packed):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        self.packed=packed;p=plan.p;n=p.n
        body=prior.source_text
        marker='CUtensorMap premap,maskmap,outmap[4];int N;'
        assert body.count(marker)==1
        body=body.replace(marker,'CUtensorMap premap,maskmap,outmap[4],lowmap,highmap;int N;')
        marker='  if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  if constexpr(PLANE<2){
   tma_load_2d(sm+PLANE*32768+q,&p.lowmap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
   tma_load_2d(sm+PLANE*32768+q+4096,&p.highmap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
  }''')
        marker='template<int MODE,int PLANE> TMN_DEVI void load_epi('
        helper='''
TMN_DEVI uint32_t split_decode(unsigned lo,unsigned ex){
 unsigned a=(lo&127u)|((lo&128u)<<8)|(((ex&255u)^SPLIT_XOR)<<7);
 unsigned b=((lo>>8)&127u)|(lo&32768u)|(((ex>>8)^SPLIT_XOR)<<7);
 return a|(b<<16);
}
'''
        body=body.replace(marker,helper+marker)
        marker='  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+32768+off);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  unsigned base=(WG*2+c/64)*8192+sw64(r*64+c%64);
  uint32_t gr=split_decode(*reinterpret_cast<uint16_t*>(sm+base),*reinterpret_cast<uint16_t*>(sm+base+4096));
  uint32_t pr=split_decode(*reinterpret_cast<uint16_t*>(sm+32768+base),*reinterpret_cast<uint16_t*>(sm+32768+base+4096));''')
        marker='  *reinterpret_cast<uint32_t*>(sm+off)=dp;*reinterpret_cast<uint32_t*>(sm+32768+off)=dg;'
        assert body.count(marker)==1
        body=body.replace(marker,'  v[j]=__uint_as_float(dp);v[j+1]=__uint_as_float(dg);')
        marker=' });\n}\ntemplate<int MODE> TMN_DEVI void run'
        assert body.count(marker)==1
        body=body.replace(marker,'''
 });
 // All byte-plane readers finish before expanding into their retired storage.
 __syncthreads();
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(WG*2+c/64)*8192+swz128(r,(c%64)*2);
  *reinterpret_cast<uint32_t*>(sm+off)=__float_as_uint(v[j]);
  *reinterpret_cast<uint32_t*>(sm+32768+off)=__float_as_uint(v[j+1]);
 });
}
template<int MODE> TMN_DEVI void run''')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_split_byte_gp')
        body+='''
extern "C" __global__ void mw_split_byte_decode_check(const uint16_t* low,const uint16_t* high,uint32_t* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<32768)out[i]=split_decode(low[i],high[i]);
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DSPLIT_XOR={packed.xor}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_split_byte_gp').kernel('mw_wide_split_byte_gp');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[64,64],dims=[n,8*p.D*n],strides_bytes=[n],swizzle='64B',l2='128B')
        fields=prior.params.fields.copy();fields[-1:-1]=[tm(packed.low),tm(packed.high)];self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
    def validate_decoder(self):
        basis=torch.arange(65536,device=self.p.x.device,dtype=torch.int32).to(torch.int16)
        packed=SplitBytePre(basis.view(torch.bfloat16).reshape(1,-1),self.packed.xor);packed()
        output=torch.empty((32768,),device=basis.device,dtype=torch.int32)
        k=self.k.unit.kernel('mw_split_byte_decode_check')
        k.launch((128,1,1),(256,1,1),[packed.low,packed.high,output],0)
        # Keep the allocator owner through the benchmark's empty_cache calls.
        self.decoder_owner=(packed,output,basis)
        return torch.equal(output.view(torch.int16),basis)
