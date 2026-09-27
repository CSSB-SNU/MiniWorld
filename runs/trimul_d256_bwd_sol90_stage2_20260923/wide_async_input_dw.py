"""All input-gradient weight columns share each loaded GP tile."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class AsyncInputDW:
    def __init__(self,plan):
        p=plan.p;d=p.D;self.p=p;self.splits=plan.input_splits;root=Path(__file__).resolve().parent
        body=(root/'wide_async_input_dw.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_async_input_dw').kernel('mw_wide_async_input_dw')
        self.threads=(d//128+1)*128;self.smem=3*(8192+128*d)+128;assert self.smem<=232448
        self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module()
        gp=launch.tensor_map(p.gp_all,[64,64],dims=[p.M,8*d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        xn=launch.tensor_map(p.xn,[64,64],dims=[d,p.M],strides_bytes=[d*2],swizzle='128B',l2='128B')
        self.params=launch.Struct([gp,xn,p.floats[7],p.M])
    def __call__(self):self.k.launch((self.splits*(8*self.p.D//64),1,1),(self.threads,1,1),[self.params],self.smem)
