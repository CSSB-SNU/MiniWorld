"""Recompute projection only in tiles whose normalized BF16 input changed."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class FlaggedProjection:
    def __init__(self,plan,changed):
        p=plan.p;d=p.D;h=2*d;root=Path(__file__).resolve().parent
        body=(root/'wide_flagged_projection.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_flagged_projection').kernel('mw_wide_flagged_projection')
        self.smem=49152+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        def tm(t,cols,rows):return L.tensor_map(t,[64,64],dims=[cols,rows],strides_bytes=[cols*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tensors[6],h,p.M),tm(plan.b1.wp,h,d),tm(plan.b1.proj,d,p.M),changed,p.M])
        self.grid=(p.M//64,d//128,1)
    def __call__(self):self.k.launch(self.grid,(128,1,1),[self.params],self.smem)
