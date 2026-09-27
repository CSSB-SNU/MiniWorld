"""Early mask reads for the existing D256 saved-products source."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class MaskSource:
    def __init__(self,original,mode):
        self.__dict__.update(original.__dict__)
        self.original=original;self.mode=mode
        body=mask_stage(original.source_text,mode).replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('mw_d256_b7_tma','mw_d256_mask_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.cubin),'mw_d256_mask_source').kernel('mw_d256_mask_source')
        self.smem=114816+(256 if mode=='bulk' else 0)
        self.source.set_max_dynamic_smem(self.smem)

    def source_only(self):
        self.source.launch((32*self.splits,1,1),(self.source_threads,1,1),[self.params],self.smem)

    def __call__(self):
        self.source_only();self.wide_finish();return self.p.outputs
