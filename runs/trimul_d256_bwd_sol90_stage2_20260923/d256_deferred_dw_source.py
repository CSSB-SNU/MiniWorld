"""Overlap the previous dW MMA with the next projection in one compute group."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage


class DeferredDWSource:
    def __init__(self,original):
        self.__dict__.update(original.__dict__)
        body=mask_stage(original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        marker='  });wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
  // This wait has also completed the previous iteration's dW operands.
  fence_regs(dw0);fence_regs(dw1);
  if(it>0){
   if(tid==0)tma_store_wait_read<0>();
   named_bar_sync(1,128);
   if(tid==0)mbar_arrive(bar+3+(1-slot));
  }
''')
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'
        assert body.count(marker)==1
        body=body.replace(marker,'});wgmma_commit();')
        marker=' for(int j=0;j<64;++j){'
        assert body.count(marker)==1
        body=body.replace(marker,' wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);\n if(tid==0)tma_store_wait_all();named_bar_sync(1,128);\n'+marker)
        body=body.replace('mw_d256_b7_tma','mw_d256_deferred_dw_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_d256_deferred_dw_source').kernel('mw_d256_deferred_dw_source')
        self.kernel.set_max_dynamic_smem(self.smem)

    def __call__(self):
        self.kernel.launch((32*self.splits,1,1),(self.source_threads,1,1),[self.params],self.smem)
