"""Exact compact sigmoid lookup in L1, preserving D256 shared-memory occupancy."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class GlobalSigmoidSource:
    def __init__(self,plan,policy='ca'):
        assert policy in ('ca','ldg')
        original=plan.b7;self.__dict__.update(original.__dict__)
        self.table=torch.empty(4096,device=plan.p.x.device,dtype=torch.float32)
        body=mask_stage(original.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        marker='float* part;int M;';assert body.count(marker)==1
        body=body.replace(marker,'float* part;const float* lut;int M;')
        helper='''
TMN_DEVI float source_sigmoid(uint32_t raw,const float* table){
 unsigned magnitude=raw&0x7fffu;
 if(magnitude>=0x3a00u&&magnitude<0x4200u){
  unsigned i=(raw>>15)*2048u+magnitude-0x3a00u;
  SIGMOID_READ
 }
 return math::sigmoid(__uint_as_float(raw<<16));
}
'''
        load='float v;asm("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(table+i));return v;' if policy=='ca' else 'return __ldg(table+i);'
        helper=helper.replace('SIGMOID_READ',load)
        marker='TMN_DEVI void packed_glu(';assert body.count(marker)==1
        body=body.replace(marker,helper+'\n'+marker)
        body=body.replace('uint32_t ma,uint32_t mb){','uint32_t ma,uint32_t mb,const float* table){')
        body=body.replace('packed_glu(pre,xn,sm+DERIV,ma,mb);','packed_glu(pre,xn,sm+DERIV,ma,mb,p.lut);')
        marker='float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);';assert body.count(marker)==1
        body=body.replace(marker,'float ga=source_sigmoid(gr&65535u,table),gb=source_sigmoid(gr>>16,table),pa=bf16lo(pr),pb=bf16hi(pr);')
        body=body.replace('mw_d256_b7_tma','mw_d256_global_sigmoid_source')
        body+='''
extern "C" __global__ void mw_build_source_sigmoid(float* table){
 unsigned i=blockIdx.x*256+threadIdx.x;
 unsigned raw=((i/2048)<<15)+0x3a00u+(i%2048);
 table[i]=math::sigmoid(__uint_as_float(raw<<16));
}
extern "C" __global__ void mw_check_source_sigmoid(const float* table,unsigned* errors){
 unsigned raw=blockIdx.x*256+threadIdx.x;
 if(__float_as_uint(source_sigmoid(raw,table))!=__float_as_uint(math::sigmoid(__uint_as_float(raw<<16))))atomicAdd(errors,1u);
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_global_sigmoid_source')
        self.k=unit.kernel('mw_d256_global_sigmoid_source');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        unit.kernel('mw_build_source_sigmoid').launch((16,1,1),(256,1,1),[self.table],0)
        self.test=unit.kernel('mw_check_source_sigmoid')
        fields=original.params.fields.copy();fields.insert(len(fields)-1,self.table);self.params=T._launch_module().Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def validate_table(self):
        errors=torch.zeros((),device=self.table.device,dtype=torch.int32)
        self.test.launch((256,1,1),(256,1,1),[self.table,errors],0)
        count=int(errors.item());assert count==0,count
        return count
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
