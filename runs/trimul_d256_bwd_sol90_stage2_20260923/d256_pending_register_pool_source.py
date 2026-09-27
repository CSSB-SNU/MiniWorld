"""N256 dW stays in flight while the next projection issues; exact CTA pool."""
from d256_register_layout_source import RegisterLayoutSource
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PendingRegisterPoolSource(RegisterLayoutSource):
    def __init__(self,plan,pending=True,readwait=False):
        super().__init__(plan,True,False);body=self.source_text
        if pending:
            marker='});wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
            assert body.count(marker)==1
            body=body.replace(marker,'''});wgmma_commit();
  if(it>0){
   wgmma_wait<1>();fence_regs(dw);
   if(tid==0)tma_store_wait_all();named_bar_sync(1,128);
   if(tid==0)mbar_arrive(bar+3+(1-slot));
  }
  wgmma_wait<0>();fence_regs(pre);''')
            marker='});wgmma_commit();wgmma_wait<0>();fence_regs(dw);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'
            assert body.count(marker)==1;body=body.replace(marker,'});wgmma_commit();')
            marker=' static_for<128>([&](auto jj){'
            assert body.count(marker)==1
            body=body.replace(marker,' wgmma_wait<0>();fence_regs(dw);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);\n'+marker)
        if readwait:
            body=body.replace('tma_store_wait_all();','tma_store_wait_read<0>();')
            end=body.rfind('extern "C" __global__');point=body.rfind('}',0,end)
            body=body[:point]+' if(tid==0)tma_store_wait_all();\n'+body[point:]
        body=body.replace('mw_d256_register_layout_source','mw_d256_pending_register_pool_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags)
        self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_pending_register_pool_source')
        self.k=T.load_unit(str(self.cubin),'mw_d256_pending_register_pool_source').kernel('mw_d256_pending_register_pool_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        assert self.registers==128 and self.occupancy==2 and self.local_bytes==0
