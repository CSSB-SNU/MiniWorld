"""TMA zero-fill pads 48-byte lossless blocks to warp-owned 64-byte output slabs."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
import torch
from wide_padded_adaptive_pre import PaddedAdaptivePre

class PaddedAdaptiveGP:
    def __init__(self,plan,packed):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        self.packed=packed;p=plan.p;n=p.n
        assert hasattr(packed,'storage')
        body=prior.source_text.replace('CUtensorMap premap,maskmap,outmap[4];','CUtensorMap premap,maskmap,outmap[4],basemap;').replace('bar+6,98304','bar+6,102400')
        self.smem=98432+4096
        marker='template<int MODE,int PLANE> TMN_DEVI void load_epi('
        helper='''
TMN_DEVI uint32_t split_decode(unsigned lo,unsigned ex,unsigned base,const bf* raw){
 if(base==255u)return *reinterpret_cast<const uint32_t*>(raw);
 unsigned mant=__byte_perm(lo,0u,0x4140);
 mant=(mant&0x007f007fu)|((mant&0x00800080u)<<8);
 unsigned codes=(ex&15u)|((ex&240u)<<12);
 return mant|((codes+base*0x00010001u)<<7);
}
TMN_DEVI void split_load4(void* dst,const CUtensorMap* map,uint64_t* bar,int c2,int c3){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,%3,%4}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c2),"r"(c3):"memory");
}
'''
        assert body.count(marker)==1;body=body.replace(marker,helper+marker)
        marker='if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);'
        assert body.count(marker)==1
        body=body.replace(marker,'if constexpr(PLANE<2){split_load4(sm+PLANE*32768+q,&p.premap,bar+6,(ni+64*wn)/32,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);if(wn==0)tma_load_2d(sm+98432+PLANE*2048+wm*1024,&p.basemap,bar+6,ni/8,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);}')
        start=body.index(' static_for<32>');end=body.index('\n}\ntemplate<int MODE> TMN_DEVI void run',start)
        body=body[:start]+'''
 static_for<4>([&](auto qq){constexpr int group=decltype(qq)::value;
  uint32_t dps[8],dgs[8];
  static_for<8>([&](auto jj){constexpr int t=decltype(jj)::value,j=group*16+t*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   unsigned tile=(WG*2+c/64)*8192;
   unsigned base=tile+r*128+(((c%64)/32)^((r&4)>>2))*64;
   unsigned bo=98432+WG*1024+r*16+c/8;
   unsigned lo=base+c%32,hi=base+32+(c%32)/2;
   constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
   int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
   size_t original=(size_t(rank*64+pc)*p.N+mi+64*WG+r)*p.N+ni+c;
   uint32_t gr=split_decode(*reinterpret_cast<uint16_t*>(sm+lo),sm[hi],sm[bo],p.pre+original);
   uint32_t pr=split_decode(*reinterpret_cast<uint16_t*>(sm+32768+lo),sm[32768+hi],sm[bo+2048],p.pre+original+size_t(32)*p.N*p.N);
   unsigned off=tile+swz128(r,(c%64)*2);
   uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+65536+off);
   asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
   float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   dps[t]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   dgs[t]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  });
  // The 32-column input/output slabs have identical warp ownership.
  __syncwarp();
  static_for<8>([&](auto jj){constexpr int t=decltype(jj)::value,j=group*16+t*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   unsigned off=(WG*2+c/64)*8192+swz128(r,(c%64)*2);
   *reinterpret_cast<uint32_t*>(sm+off)=dps[t];*reinterpret_cast<uint32_t*>(sm+32768+off)=dgs[t];
  });
 });
'''+body[end:]
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_padded_adaptive_gp')
        body+='''
extern "C" __global__ void mw_padded_adaptive_decode_check(const uint8_t* data,const uint8_t* bases,const bf* raw,uint32_t* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<32768){size_t base=size_t(i/16)*48;int local=i%16;
  out[i]=split_decode(*reinterpret_cast<const uint16_t*>(data+base+local*2),data[base+32+local],bases[i/4],raw+2*i);
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_padded_adaptive_gp').kernel('mw_wide_padded_adaptive_gp');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=prior.params.fields.copy()
        fields[16]=L.tensor_map(packed.storage,[16,4,2,64],dims=[16,3,n//32,8*p.D*n],strides_bytes=[16,48,n*3//2],swizzle='none',l2='128B')
        fields[-1:-1]=[L.tensor_map(packed.bases,[16,64],dims=[n//8,8*p.D*n],strides_bytes=[n//8],swizzle='none',l2='128B')]
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
    def validate_decoder(self):
        basis=torch.arange(65536,device=self.p.x.device,dtype=torch.int32).to(torch.int16)
        packed=PaddedAdaptivePre(basis.view(torch.bfloat16).reshape(1,-1));packed()
        output=torch.empty((32768,),device=basis.device,dtype=torch.int32)
        self.k.unit.kernel('mw_padded_adaptive_decode_check').launch((128,1,1),(256,1,1),[packed.storage,packed.bases,packed.pre,output],0)
        return torch.equal(output.view(torch.int16),basis)
