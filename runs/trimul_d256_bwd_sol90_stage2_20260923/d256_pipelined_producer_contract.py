"""Keep one ordered MMA group pending in the two-slot one-consumer pipeline."""
from d256_one_group_producer_contract import OneGroupProducerContract
from two_role_modes_register_pool import initial_pool_modes
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PipelinedProducerContract(OneGroupProducerContract):
    def __init__(self,plan,consumer=224):
        super().__init__(plan,consumer);producer=256-consumer;body=self.source_text
        marker='  fence_regs(v0);fence_regs(v1);wgmma_fence();';assert body.count(marker)==1
        body=body.replace(marker,'  if(step==0){fence_regs(v0);fence_regs(v1);wgmma_fence();}')
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(wg+1,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);'
        assert body.count(marker)==1
        body=body.replace(marker,''' });wgmma_commit();
  if(step>=1){wgmma_wait<1>();named_bar_sync(wg+1,128);if(tid==0)mbar_arrive(bar+SLOTS+(step-1)%SLOTS);}''')
        marker=' }\n named_bar_sync(1,128);\n static_for<';assert body.count(marker)==1
        body=body.replace(marker,''' }
 wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(1,128);
 if(tid==0)mbar_arrive(bar+SLOTS+5%SLOTS);
 static_for<''')
        body=body.replace('mw_d256_one_group_producer_contract','mw_d256_pipelined_producer_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DROW_GROUPS=1','-DGRID_ORDER=1','-DMIN_BLOCKS=1','-DCONTRACT_SLOTS=2']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_modes(cubin,'mw_d256_pipelined_producer_contract',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_pipelined_producer_contract').kernel('mw_d256_pipelined_producer_contract');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
