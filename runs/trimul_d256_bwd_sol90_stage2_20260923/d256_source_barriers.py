"""Test redundant warpgroup joins around already-complete async operations."""
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SourceBarriers:
    def __init__(self,plan,variant):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);body=prior.source_text
        assert 1<=variant<=7
        if variant&1:
            marker='named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);'
            assert body.count(marker)==1
            body=body.replace(marker,'fence_proxy_async();named_bar_sync(1,128);')
        if variant&2:
            marker='mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);'
            assert body.count(marker)==1
            body=body.replace(marker,'mbar_wait(bar+slot,(it/2)&1);')
        if variant&4:
            marker='if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'
            assert body.count(marker)==1
            body=body.replace(marker,'if(tid==0){tma_store_wait_all();mbar_arrive(bar+3+slot);}')
        body=body.replace('mw_d256_register_budget_source','mw_d256_source_barriers');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_source_barriers',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_source_barriers').kernel('mw_d256_source_barriers');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
