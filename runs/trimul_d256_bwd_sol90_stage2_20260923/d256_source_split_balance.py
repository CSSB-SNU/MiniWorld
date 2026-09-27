"""Vary input-source CTA count, folding FP32 input partials into the existing eight."""
import torch
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SplitBalanceSource:
    def __init__(self,plan,splits=4):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__)
        assert (plan.p.D,plan.p.n,prior.splits)==(256,384,8)
        assert splits in (4,6,12,16);self.splits=splits
        p=plan.p;self.partial=torch.empty((splits,11*256*256),device=p.x.device,dtype=torch.float32)
        body=prior.source_text.replace('mw_d256_register_budget_source','mw_d256_split_balance_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_split_balance_source',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_split_balance_source').kernel('mw_d256_split_balance_source');self.k.set_max_dynamic_smem(self.smem)
        fields=prior.params.fields.copy();fields[-2]=self.partial;self.params=T._launch_module().Struct(fields)
        reduce_body='''extern "C" __global__ void mw_fold_source_partials(const float* input,float* output){
 constexpr int D=256,PART=8*D*D,STRIDE=11*D*D;
 for(int ix=blockIdx.x*256+threadIdx.x;ix<8*PART;ix+=gridDim.x*256){
  int group=ix/PART,q=ix%PART;float value=0.f;
  for(int s=group*WEIGHT_SPLITS/8;s<(group+1)*WEIGHT_SPLITS/8;++s)value+=input[s*STRIDE+3*D*D+q];
  output[group*STRIDE+3*D*D+q]=value;
 }
}
'''
        self.fold_cubin=T.compile_text(reduce_body,flags);self.fold=T.load_unit(str(self.fold_cubin),'mw_fold_source_partials').kernel('mw_fold_source_partials');self.output=p.floats[7]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):
        self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
        self.fold.launch((528,1,1),(256,1,1),[self.partial,self.output],0)
