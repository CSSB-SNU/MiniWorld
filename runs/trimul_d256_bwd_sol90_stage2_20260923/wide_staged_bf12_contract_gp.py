"""Expand compact preactivation tiles in reverse order, then reuse GP math."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_bf12_contract_gp import PackedContractGP

class StagedPackedContractGP(PackedContractGP):
    def __init__(self,plan,packed):
        super().__init__(plan,packed)
        body=self.source_text
        body=body.replace('tma_load_2d(sm+49152+q,&p.maskmap','tma_load_2d(sm+65536+q,&p.maskmap')
        body=body.replace('sm+t*4096,&p.mantmap','sm+t*6144,&p.mantmap')
        body=body.replace('sm+16384+t*2048,&p.expmap','sm+t*6144+4096,&p.expmap')
        body=body.replace('sm+24576+t*4096,&p.mantmap','sm+24576+t*6144,&p.mantmap')
        body=body.replace('sm+40960+t*2048,&p.expmap','sm+24576+t*6144+4096,&p.expmap')
        helper='''
template<int MODE,int PLANE> TMN_DEVI void expand_pre(const Params& p,uint8_t* sm,int ch,int mi,int ni){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 #pragma unroll 1
 for(int t=3;t>=0;--t){uint32_t decoded[8];
  #pragma unroll
  for(int q=0;q<8;++q){int i=threadIdx.x+q*256,r=i/32,c=(i%32)*2;
   unsigned mo=sw64(r*64+c),eo=packed_sw32(r*32+c/2);
   unsigned mant=*reinterpret_cast<uint16_t*>(sm+PLANE*24576+t*6144+mo);
   unsigned codes=sm[PLANE*24576+t*6144+4096+eo];
   size_t original=(size_t(rank*64+pc+32*PLANE)*p.N+mi+(t/2)*64+r)*p.N+ni+(t%2)*64+c;
   decoded[q]=decode_pair(mant,codes,p.pre+original);
  }
  __syncthreads();
  #pragma unroll
  for(int q=0;q<8;++q){int i=threadIdx.x+q*256,r=i/32,c=(i%32)*2;
   *reinterpret_cast<uint32_t*>(sm+PLANE*32768+t*8192+swz128(r,c*2))=decoded[q];
  }
  __syncthreads();
 }
}
'''
        marker='template<int MODE> TMN_DEVI void consume('
        assert body.count(marker)==1;body=body.replace(marker,helper+'\n'+marker)
        begin=body.index(' static_for<32>');end=body.index('\n}\ntemplate<int MODE> TMN_DEVI void run',begin)
        original=plan.contract_gp.source_text
        ob=original.index(' static_for<32>');oe=original.index('\n}\ntemplate<int MODE> TMN_DEVI void run',ob)
        body=body[:begin]+' expand_pre<MODE,1>(p,sm,ch,mi,ni);expand_pre<MODE,0>(p,sm,ch,mi,ni);\n'+original[ob:oe]+body[end:]
        body=body.replace('mw_wide_bf12_contract_gp','mw_wide_staged_bf12_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.source_text=body
        self.k=T.load_unit(str(self.cubin),'mw_wide_staged_bf12_contract_gp').kernel('mw_wide_staged_bf12_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fun=drv.d.CUfunction(int(self.k.handle))
        attr=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fun)))
        self.registers=attr('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=attr('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
