"""Keep dW pending while the same consumer issues its next projection."""
from d256_deep_pair_source import DeepPairSource
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class AsyncPairSource(DeepPairSource):
    def __init__(self,p,old,full_n=False,phase=0,slots=3):
        assert phase in (0,1)
        super().__init__(p,old,full_n,phase,slots)
        body=self.source_text
        pre='  wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
        assert body.count(pre)==1
        body=body.replace(pre,'''  wgmma_commit();
  if(it>0){
   wgmma_wait<1>();named_bar_sync(SYNC,128);
   if(tid==0)mbar_arrive(b+PAIR_SLOTS+1+(it-1)%PAIR_SLOTS);
  }
  wgmma_wait<0>();fence_regs(pre);''')
        old_tail='wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();named_bar_sync(SYNC,128);if(tid==0)mbar_arrive(b+PAIR_SLOTS+1+slot);'
        assert body.count(old_tail)==1
        body=body.replace(old_tail,'wgmma_commit();if(tid==0)tma_store_wait_all();named_bar_sync(SYNC,128);')
        marker=' }\n static_for<64>([&](auto jj)'
        assert body.count(marker)==1
        body=body.replace(marker,''' }
 wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);named_bar_sync(SYNC,128);
 if(tid==0)mbar_arrive(b+PAIR_SLOTS+1+(end-begin-1)%PAIR_SLOTS);
 static_for<64>([&](auto jj)''')
        body=body.replace('mw_d256_deep_pair_source','mw_d256_async_pair_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DPAIR_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_async_pair_source').kernel('mw_d256_async_pair_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2),'insufficient dynamic register pool'
