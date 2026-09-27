"""Offset alternate source CTAs on each SM to test tensor/GLU phase overlap."""
import torch
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class StaggeredCTASource:
    def __init__(self,plan,cycles=1000):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);body=prior.source_text
        self.counter=torch.empty(256,device=plan.p.x.device,dtype=torch.int32)
        marker='float* part;int M;';assert body.count(marker)==1
        body=body.replace(marker,'float* part;unsigned* sm_counter;int M;')
        marker=' __syncthreads();\n if(threadIdx.x<128)';assert body.count(marker)==1
        body=body.replace(marker,f''' __syncthreads();
 if(threadIdx.x==128){{
  unsigned smid;asm volatile("mov.u32 %0,%smid;":"=r"(smid));
  unsigned slot=atomicAdd(p.sm_counter+smid,1u);
  if(slot&1u){{unsigned long long begin=clock64();while(clock64()-begin<{cycles}ull)__nanosleep(32);}}
 }}
 __syncthreads();
 if(threadIdx.x<128)''')
        body=body.replace('mw_d256_register_budget_source','mw_d256_staggered_cta_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_staggered_cta_source',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_staggered_cta_source').kernel('mw_d256_staggered_cta_source');self.k.set_max_dynamic_smem(self.smem)
        fields=prior.params.fields.copy();fields.insert(-1,self.counter);self.params=T._launch_module().Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):
        self.counter.zero_();self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
