"""Overlap producer GP stores with the next projection before input-slot reuse."""
from d256_whole_aliased_source import WholeAliasedSource
from two_role_register_pool import initial_pool_roles
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class DelayedStoreSource:
    def __init__(self,plan,producer=96,read_only=False):
        prior=WholeAliasedSource(plan,producer);self.__dict__.update(prior.__dict__)
        consumer=256-producer;body=prior.source_text
        marker='tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();'
        assert body.count(marker)==1
        body=body.replace(marker,'tma_store_commit();mbar_arrive(bar+3+slot);')
        marker='   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);'
        assert body.count(marker)==1
        wait='tma_store_wait_read<0>();' if read_only else 'tma_store_wait_all();'
        body=body.replace(marker,'   if(tid==0){if(it>=1)'+wait+'load_input(p,sm,bar,(tile+1)*64,rank,1-slot);}')
        start=body.index('TMN_DEVI void produce(');end=body.index('template<int OA,int OB,int TA,int TB> TMN_DEVI void mma256_off',start)
        fragment=body[start:end]
        marker='\n }\n}\n'
        assert fragment.count(marker)==1
        fragment=fragment.replace(marker,'\n }\n if(tid==0)tma_store_wait_all();\n named_bar_sync(1,128);\n}\n')
        body=body[:start]+fragment+body[end:]
        body=body.replace('mw_d256_whole_aliased_source','mw_d256_delayed_store_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_roles(cubin,'mw_d256_delayed_store_source',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_delayed_store_source').kernel('mw_d256_delayed_store_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
