"""Wide native ordered GEMM plus input LN; no prefix or dXn HBM intermediate."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W

class FusedDx:
    def __init__(self,p):
        root=Path(__file__).resolve().parent;pre=root.parent/'trimul_d256_bwd_sol90_20260923'
        d=p.D
        body=(root/'wide_fused_dx.cu').read_text().replace('// MMA_HELPERS',(pre/'mma.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_fused_dx').kernel('mw_wide_fused_dx')
        self.smem=max(3*d*128,2*(d//64+1)*8192)+128
        self.k.set_max_dynamic_smem(self.smem);drv=self.k.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),384,self.smem)))
        nr=int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(drv.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,drv.d.CUfunction(int(self.k.handle)))))
        assert nr*3>=496,(nr,occ)
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module();t7=p.tensors.copy();t7[22:24]=[p.dl,p.dr]
        self.params=L.Struct([*p.maps7,*t7,*p.floats,p.M,p.n,W.tm(p.x.reshape(p.M,d)),W.tm(p.dy.reshape(p.M,d)),W.tm(p.dx.reshape(p.M,d))])
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,384,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
