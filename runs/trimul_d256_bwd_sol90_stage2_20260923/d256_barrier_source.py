"""Use per-thread mbarrier acquire and one publication barrier for local GP."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class BarrierSource:
    def __init__(self,plan,mode='both'):
        prior=plan.b7;self.__dict__.update(prior.__dict__)
        assert mode in ('input','proxy','both')
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        if mode in ('input','both'):
            for a,b in (
                ('mbar_wait(bar+2,0);named_bar_sync(1,128);','mbar_wait(bar+2,0);'),
                ('mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);','mbar_wait(bar+slot,(it/2)&1);')):
                assert body.count(a)==1;body=body.replace(a,b)
        if mode in ('proxy','both'):
            a='named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);'
            assert body.count(a)==1
            body=body.replace(a,'fence_proxy_async();named_bar_sync(1,128);')
        body=body.replace('mw_d256_b7_tma','mw_d256_barrier_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_barrier_source').kernel('mw_d256_barrier_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
