"""Explicit native output-projection dW with independent FP32 split storage."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class NativeProjectionDW:
    def __init__(self,plan,splits=16):
        p=plan.p;d=p.D;h=2*d;assert d in (256,384,512) and p.M%(64*splits)==0
        root=Path(__file__).resolve().parent
        body=(root/'wide_native_projection_dw.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DDW_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_native_projection_dw')
        self.k=unit.kernel('mw_wide_native_projection_dw');self.reduce=unit.kernel('mw_wide_native_projection_dw_reduce')
        self.smem=98304+128;self.grid=(d//128)*(h//128)*splits;self.reduce_grid=(d*h+511)//512
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module()
        self.partial=torch.empty((splits,d,h),device=p.x.device,dtype=torch.float32)
        a=L.tensor_map(p.tensors[7],[64,64,2],dims=[64,p.M,d//64],strides_bytes=[d*2,128],swizzle='128B',l2='256B')
        b=L.tensor_map(p.tensors[6],[64,64,2],dims=[64,p.M,h//64],strides_bytes=[h*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(self.partial,[32,64,4],dims=[32,d*splits,h//32],strides_bytes=[h*4,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y,self.partial,p.dwp,p.M]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
    def __call__(self):
        self.k.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
        self.reduce.launch((self.reduce_grid,1,1),(256,1,1),[self.params],0)
