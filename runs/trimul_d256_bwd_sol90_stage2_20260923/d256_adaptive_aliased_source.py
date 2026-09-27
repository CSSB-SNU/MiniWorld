"""Defer only unavailable input credits until after publishing the current GP."""
from d256_whole_aliased_source import WholeAliasedSource
from two_role_register_pool import initial_pool_roles
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class AdaptiveAliasedSource(WholeAliasedSource):
    def __init__(self,plan,producer=80,pending=False):
        super().__init__(plan,producer);consumer=256-producer
        body=self.source_text
        helper='''TMN_DEVI bool credit_ready(uint64_t* bar,int phase){
 unsigned ok;asm volatile("{.reg .pred P;mbarrier.test_wait.parity.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(ok):"r"(smem_u32(bar)),"r"(phase):"memory");return ok;
}
'''
        body=body.replace('TMN_DEVI void produce(',helper+'TMN_DEVI void produce(')
        old='''  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }'''
        assert body.count(old)==1
        body=body.replace(old,'''  bool next_loaded=false;
  if(tid==0 && tile+1<end && (it<1 || credit_ready(bar+5+(1-slot),((it-1)/2)&1))){
   load_input(p,sm,bar,(tile+1)*64,rank,1-slot);next_loaded=true;
  }''')
        old='tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();';assert body.count(old)==1
        body=body.replace(old,'''tma_store_commit();mbar_arrive(bar+3+slot);
   if(tile+1<end && !next_loaded){
    if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
    load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
   }
   tma_store_wait_all();''')
        if pending:
            old='''});wgmma_commit();wgmma_wait<0>();fence_regs(dw);
  named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+5+slot);
 }'''
            assert body.count(old)==1
            body=body.replace(old,''' });wgmma_commit();
  if(it>=1){wgmma_wait<1>();named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+5+(1-slot));}
 }
 wgmma_wait<0>();fence_regs(dw);named_bar_sync(2,128);
 if(tid==0)mbar_arrive(bar+5+(end-begin-1)%2);''')
            # The accumulator remains asynchronous between iterations.
            body=body.replace('  fence_regs(dw);wgmma_fence();','  if(it==0){fence_regs(dw);wgmma_fence();}')
        body=body.replace('mw_d256_whole_aliased_source','mw_d256_adaptive_aliased_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_roles(cubin,'mw_d256_adaptive_aliased_source',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_adaptive_aliased_source').kernel('mw_d256_adaptive_aliased_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
