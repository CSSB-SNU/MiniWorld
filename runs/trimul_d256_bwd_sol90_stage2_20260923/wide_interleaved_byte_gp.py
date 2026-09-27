"""One 4D TMA interleaves low/high byte slabs so each warp can decode in place."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_split_byte_gp import SplitByteGP

class InterleavedByteGP(SplitByteGP):
    def __init__(self,plan,packed):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        self.packed=packed;p=plan.p;n=p.n
        assert hasattr(packed,'storage')
        body=prior.source_text
        marker='template<int MODE,int PLANE> TMN_DEVI void load_epi('
        helper='''
TMN_DEVI uint32_t split_decode(unsigned lo,unsigned ex){
 unsigned a=(lo&127u)|((lo&128u)<<8)|(((ex&255u)^SPLIT_XOR)<<7);
 unsigned b=((lo>>8)&127u)|(lo&32768u)|(((ex>>8)^SPLIT_XOR)<<7);
 return a|(b<<16);
}
TMN_DEVI void split_load4(void* dst,const CUtensorMap* map,uint64_t* bar,int c2,int c3){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,%3,%4}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c2),"r"(c3):"memory");
}
'''
        assert body.count(marker)==1;body=body.replace(marker,helper+marker)
        marker='if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);'
        assert body.count(marker)==1
        body=body.replace(marker,'if constexpr(PLANE<2)split_load4(sm+PLANE*32768+q,&p.premap,bar+6,(ni+64*wn)/16,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);')
        start=body.index(' static_for<32>');end=body.index('\n}\ntemplate<int MODE> TMN_DEVI void run',start)
        body=body[:start]+'''
 static_for<8>([&](auto qq){constexpr int group=decltype(qq)::value;
  uint32_t dps[4],dgs[4];
  static_for<4>([&](auto jj){constexpr int t=decltype(jj)::value,j=group*8+t*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   unsigned tile=(WG*2+c/64)*8192;
   unsigned byte=(c%64)/16*32+c%16;
   unsigned lo=tile+r*128+byte,hi=tile+r*128+byte+16;
   uint32_t gr=split_decode(*reinterpret_cast<uint16_t*>(sm+lo),*reinterpret_cast<uint16_t*>(sm+hi));
   uint32_t pr=split_decode(*reinterpret_cast<uint16_t*>(sm+32768+lo),*reinterpret_cast<uint16_t*>(sm+32768+hi));
   unsigned off=tile+swz128(r,(c%64)*2);
   uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+65536+off);
   asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
   float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   dps[t]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   dgs[t]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  });
  // The 16-column input/output slabs have identical warp ownership.
  __syncwarp();
  static_for<4>([&](auto jj){constexpr int t=decltype(jj)::value,j=group*8+t*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   unsigned off=(WG*2+c/64)*8192+r*128+(c%64)*2;
   *reinterpret_cast<uint32_t*>(sm+off)=dps[t];*reinterpret_cast<uint32_t*>(sm+32768+off)=dgs[t];
  });
 });
'''+body[end:]
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_interleaved_byte_gp')
        body+='''
extern "C" __global__ void mw_split_byte_decode_check(const uint16_t* low,const uint16_t* high,uint32_t* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<32768)out[i]=split_decode(low[i],high[i]);
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DSPLIT_XOR={packed.xor}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_interleaved_byte_gp').kernel('mw_wide_interleaved_byte_gp');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=prior.params.fields.copy()
        fields[16]=L.tensor_map(packed.storage,[16,2,4,64],dims=[16,2,n//16,8*p.D*n],strides_bytes=[packed.pre.numel(),16,n],swizzle='none',l2='128B')
        for i,g in enumerate(p.gp):
            fields[18+i]=L.tensor_map(g,[64,64],dims=[n,2*p.D*n],strides_bytes=[n*2],swizzle='none',l2='128B')
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
