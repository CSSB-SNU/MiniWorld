"""Overlap the selected source's GP store with its next independent projection."""
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class OverlappedStoreSource:
    def __init__(self,plan,read_only=False):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);body=prior.source_text
        marker='if(tid==0)tma_store_wait_all();';assert body.count(marker)==1
        body=body.replace(marker,'')
        marker='wgmma_wait<0>();fence_regs(pre);';assert body.count(marker)==1
        wait='tma_store_wait_read<0>();' if read_only else 'tma_store_wait_all();'
        body=body.replace(marker,marker+'\n  if(tid==0 && it>0)'+wait+'\n  named_bar_sync(1,128);')
        marker=' static_for<128>([&](auto jj){constexpr int j=decltype(jj)::value;'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid==0)tma_store_wait_all();\n named_bar_sync(1,128);\n'+marker)
        body=body.replace('mw_d256_register_budget_source','mw_d256_overlapped_store_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_overlapped_store_source',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_overlapped_store_source').kernel('mw_d256_overlapped_store_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
