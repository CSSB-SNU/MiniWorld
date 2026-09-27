"""Branch-free exponent reconstruction, with an almost-zero compressed high plane."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_exponent_nibble_pre import ExponentNibblePre

class ExponentNibbleGP:
    def __init__(self,plan,packed):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        self.packed=packed;p=plan.p;n=p.n
        body=prior.source_text
        marker='CUtensorMap premap,maskmap,outmap[4];int N;'
        assert body.count(marker)==1
        body=body.replace(marker,'CUtensorMap premap,maskmap,outmap[4],lowmap,elomap,ehimap;int N;')
        marker='  if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  if constexpr(PLANE<2){
   int t=wm*2+wn,pr=(rank*64+pc+PLANE*32)*p.N+mi+64*wm;
   tma_load_2d(sm+PLANE*32768+t*4096,&p.lowmap,bar+6,ni+64*wn,pr);
   tma_load_2d(sm+PLANE*32768+16384+t*2048,&p.elomap,bar+6,(ni+64*wn)/2,pr);
   tma_load_2d(sm+PLANE*32768+24576+t*2048,&p.ehimap,bar+6,(ni+64*wn)/2,pr);
  }''')
        marker='template<int MODE,int PLANE> TMN_DEVI void load_epi('
        helper='''
TMN_DEVI unsigned nibble_sw32(unsigned v){return v^(((v>>7)&1u)<<4);}
TMN_DEVI uint32_t split_decode(unsigned lo,unsigned elo,unsigned ehi){
 unsigned ea=(((elo&15u)|((ehi&15u)<<4))+116u)&255u;
 unsigned eb=(((elo>>4)|(ehi&240u))+116u)&255u;
 unsigned a=(lo&127u)|((lo&128u)<<8)|(ea<<7);
 unsigned b=((lo>>8)&127u)|(lo&32768u)|(eb<<7);
 return a|(b<<16);
}
'''
        body=body.replace(marker,helper+marker)
        marker='  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+32768+off);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  unsigned t=WG*2+c/64,mo=t*4096+sw64(r*64+c%64),eo=t*2048+nibble_sw32(r*32+(c%64)/2);
  uint32_t gr=split_decode(*reinterpret_cast<uint16_t*>(sm+mo),sm[16384+eo],sm[24576+eo]);
  uint32_t pr=split_decode(*reinterpret_cast<uint16_t*>(sm+32768+mo),sm[49152+eo],sm[57344+eo]);''')
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
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_exponent_nibble_gp')
        body+='''
extern "C" __global__ void mw_split_byte_decode_check(const uint16_t* low,const uint8_t* elo,const uint8_t* ehi,uint32_t* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<32768)out[i]=split_decode(low[i],elo[i],ehi[i]);
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_exponent_nibble_gp').kernel('mw_wide_exponent_nibble_gp');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[64,64],dims=[n,8*p.D*n],strides_bytes=[n],swizzle='64B',l2='128B')
        nib=lambda t:L.tensor_map(t,[32,64],dims=[n//2,8*p.D*n],strides_bytes=[n//2],swizzle='32B',l2='128B')
        fields=prior.params.fields.copy();fields[-1:-1]=[tm(packed.low),nib(packed.explo),nib(packed.exphi)];self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
    def validate_decoder(self):
        basis=torch.arange(65536,device=self.p.x.device,dtype=torch.int32).to(torch.int16)
        packed=ExponentNibblePre(basis.view(torch.bfloat16).reshape(1,-1));packed()
        output=torch.empty((32768,),device=basis.device,dtype=torch.int32)
        k=self.k.unit.kernel('mw_split_byte_decode_check')
        k.launch((128,1,1),(256,1,1),[packed.low,packed.explo,packed.exphi,output],0)
        # Keep the allocator owner through the benchmark's empty_cache calls.
        self.decoder_owner=(packed,output,basis)
        return torch.equal(output.view(torch.int16),basis)
