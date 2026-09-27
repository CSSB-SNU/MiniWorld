"""Initialize the projection accumulator through the first WGMMA's output operand."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class OutputInitSource:
    def __init__(self,plan):
        prior=plan.b7;self.__dict__.update(prior.__dict__)
        root=Path(__file__).resolve().parent
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        helper=(root/'mma_offset.cuh').read_text().split('template<int OA,int OB,int TA,int TB> TMN_DEVI void mma128_off')[0]
        helper=helper.replace('mma64_off','mma64_init').replace('"+f"','"=f"')
        marker='TMN_DEVI void compute_source('
        assert body.count(marker)==1;body=body.replace(marker,helper+'\n'+marker)
        old='float pre[32]={};fence_regs(pre);wgmma_fence();'
        assert body.count(old)==1;body=body.replace(old,'float pre[32];wgmma_fence();')
        old='mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);'
        assert body.count(old)==1
        body=body.replace(old,'if constexpr(k==0)mma64_init<0,0,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),0);else '+old)
        body=body.replace('mw_d256_b7_tma','mw_d256_output_init_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_output_init_source').kernel('mw_d256_output_init_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
