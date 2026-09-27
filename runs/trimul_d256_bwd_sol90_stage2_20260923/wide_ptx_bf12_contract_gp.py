"""Keep lossless BF12 decoder temporaries inside predicated PTX blocks."""
from wide_bf12_contract_gp import PackedContractGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PtxPackedContractGP(PackedContractGP):
    def __init__(self,plan,packed):
        super().__init__(plan,packed);body=self.source_text
        begin=body.index('TMN_DEVI uint32_t decode_pair(')
        end=body.index('\n}',begin)+2
        body=body[:begin]+'''
TMN_DEVI uint32_t decode_pair(unsigned mant,unsigned codes,const bf* original){
 uint32_t result;
 asm volatile("{\\n"
  ".reg .b32 a,b,c,e,s; .reg .b64 addr; .reg .pred zero,escape;\\n"
  "and.b32 c,%2,15; and.b32 a,%1,127; and.b32 s,%1,128; shl.b32 s,s,8; or.b32 a,a,s;\\n"
  "setp.eq.u32 zero,c,0; add.u32 e,c,115; selp.u32 e,0,e,zero; shl.b32 e,e,7; or.b32 a,a,e;\\n"
  "setp.eq.u32 escape,c,15; @escape ld.global.u16 a,[%3];\\n"
  "shr.u32 c,%2,4; shr.u32 b,%1,8; and.b32 b,b,127; and.b32 s,%1,32768; or.b32 b,b,s;\\n"
  "setp.eq.u32 zero,c,0; add.u32 e,c,115; selp.u32 e,0,e,zero; shl.b32 e,e,7; or.b32 b,b,e;\\n"
  "setp.eq.u32 escape,c,15; add.u64 addr,%3,2; @escape ld.global.u16 b,[addr];\\n"
  "shl.b32 b,b,16; or.b32 %0,a,b;\\n}"
  :"=r"(result):"r"(mant),"r"(codes),"l"(original):"memory");
 return result;
}
'''+body[end:]
        body=body.replace('mw_wide_bf12_contract_gp','mw_wide_ptx_bf12_contract_gp')
        body+='''
extern "C" __global__ void mw_check_bf12_decode(const bf* original,const uint8_t* mant,const uint8_t* codes,uint32_t* out,int pairs){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<pairs)out[i]=decode_pair(*reinterpret_cast<const uint16_t*>(mant+2*i),codes[i],original+2*i);
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_ptx_bf12_contract_gp').kernel('mw_wide_ptx_bf12_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def validate_decoder(self):
        import torch
        from wide_bf12_contract_gp import PackedPre
        basis=torch.arange(65536,device=self.p.x.device,dtype=torch.int32).to(torch.int16)
        tiny=PackedPre(basis.view(torch.bfloat16).reshape(1,-1));tiny()
        restored=torch.empty((32768,),device=basis.device,dtype=torch.int32)
        kernel=self.k.unit.kernel('mw_check_bf12_decode');L=T._launch_module()
        kernel.launch((128,1,1),(256,1,1),[tiny.pre,tiny.mant,tiny.exps,restored,L.i32(32768)],0)
        return torch.equal(restored.view(torch.int16),basis)
