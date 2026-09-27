"""Exact sigmoid using tanh plus a packed four-bit ULP correction table."""
import torch
from wide_mask_transform import mask_stage
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class CorrectedSigmoidSource:
    def __init__(self,plan,base=0x3c80):
        original=plan.b7;self.splits=original.splits
        self.table=torch.empty(256,device=plan.p.x.device,dtype=torch.int32)
        body=mask_stage(original.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        marker='float* part;int M;';assert body.count(marker)==1
        body=body.replace(marker,'float* part;const unsigned* lut;int M;')
        helper=r'''
TMN_DEVI float correction_approx(unsigned raw){
 float g=__uint_as_float(raw<<16),t;
 asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(__fmul_rn(g,0.5f)));
 return __fmaf_rn(t,0.5f,0.5f);
}
TMN_DEVI float source_sigmoid(unsigned raw,const unsigned* table){
 unsigned mag=raw&0x7fffu;
 if(mag>=LUT_BASE && mag<LUT_BASE+1024u){
  unsigned ix=(raw>>15)*1024u+mag-LUT_BASE;
  unsigned nib=(table[ix>>3]>>((ix&7u)*4u))&15u;
  if(nib!=8u){
   int delta=int(nib<<28)>>28;
   return __uint_as_float(__float_as_uint(correction_approx(raw))+delta);
  }
 }
 return math::sigmoid(__uint_as_float(raw<<16));
}
'''
        marker='TMN_DEVI void packed_glu(';assert body.count(marker)==1
        body=body.replace(marker,helper+'\n'+marker)
        body=body.replace('uint32_t ma,uint32_t mb){','uint32_t ma,uint32_t mb,const unsigned* table){')
        body=body.replace('packed_glu(pre,xn,sm+DERIV,ma,mb);','packed_glu(pre,xn,sm+DERIV,ma,mb,reinterpret_cast<unsigned*>(sm+115072));')
        marker='float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);';assert body.count(marker)==1
        body=body.replace(marker,'float ga=source_sigmoid(gr&65535u,table),gb=source_sigmoid(gr>>16,table),pa=bf16lo(pr),pb=bf16hi(pr);')
        marker=' if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();}'
        assert body.count(marker)==1
        body=body.replace(marker,' reinterpret_cast<unsigned*>(sm+115072)[threadIdx.x]=p.lut[threadIdx.x];\n'+marker)
        body=body.replace('mw_d256_b7_tma','mw_d256_corrected_sigmoid_source')
        body+=r'''
extern "C" __global__ void mw_build_correction(unsigned* table){
 unsigned word=threadIdx.x,value=0;
 #pragma unroll
 for(unsigned j=0;j<8;++j){
  unsigned ix=word*8+j,raw=((ix/1024)<<15)+LUT_BASE+(ix%1024);
  int delta=int(__float_as_uint(math::sigmoid(__uint_as_float(raw<<16))))-int(__float_as_uint(correction_approx(raw)));
  unsigned nib=(delta>=-7&&delta<=7)?(unsigned(delta)&15u):8u;
  value|=nib<<(j*4);
 }
 table[word]=value;
}
extern "C" __global__ void mw_check_correction(const unsigned* table,unsigned* errors){
 unsigned raw=blockIdx.x*256+threadIdx.x;
 if(__float_as_uint(source_sigmoid(raw,table))!=__float_as_uint(math::sigmoid(__uint_as_float(raw<<16))))atomicAdd(errors,1u);
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DLUT_BASE={base}u']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_corrected_sigmoid_source')
        self.k=unit.kernel('mw_d256_corrected_sigmoid_source');self.smem=116096;self.k.set_max_dynamic_smem(self.smem)
        unit.kernel('mw_build_correction').launch((1,1,1),(256,1,1),[self.table],0)
        self.test=unit.kernel('mw_check_correction')
        fields=original.params.fields.copy();fields.insert(len(fields)-1,self.table);self.params=T._launch_module().Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        assert self.registers*256>=128*(32+224),'insufficient dynamic register pool'

    def validate_table(self):
        errors=torch.zeros((),device=self.table.device,dtype=torch.int32)
        self.test.launch((256,1,1),(256,1,1),[self.table,errors],0)
        count=int(errors.item());assert count==0,count
        words=self.table.cpu().tolist()
        return dict(mismatches=count,escapes=sum(((w>>(j*4))&15)==8 for w in words for j in range(8)),patterns=65536)

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
