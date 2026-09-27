"""Issue the next preactivation while the preceding local dW is in flight."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class PendingDwSource:
    def __init__(self,plan,readwait=False):
        prior=plan.b7;self.__dict__.update(prior.__dict__)
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        old='});wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
        assert body.count(old)==1
        body=body.replace(old,'''});wgmma_commit();
  if(it>0){
   wgmma_wait<1>();fence_regs(dw0);fence_regs(dw1);
   if(tid==0)tma_store_wait_all();
   named_bar_sync(1,128);
   if(tid==0)mbar_arrive(bar+3+(1-slot));
  }
  wgmma_wait<0>();fence_regs(pre);''')
        old='});wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'
        assert body.count(old)==1
        body=body.replace(old,'});wgmma_commit();')
        marker=' for(int j=0;j<64;++j){'
        assert body.count(marker)==1
        body=body.replace(marker,''' wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);
 if(tid==0)tma_store_wait_all();named_bar_sync(1,128);
'''+marker)
        if readwait:
            body=body.replace('if(tid==0)tma_store_wait_all();\n   named_bar_sync','if(tid==0)tma_store_wait_read<0>();\n   named_bar_sync')
        body=body.replace('mw_d256_b7_tma','mw_d256_pending_dw_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_pending_dw_source').kernel('mw_d256_pending_dw_source')
        self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
