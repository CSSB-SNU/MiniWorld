"""Standalone D256 full-width dX from the exact ordered prefix tensor."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FullWidthPrefixDX:
    def __init__(self,plan,slots=3,k_tile=64,depth=1,l2='128B'):
        p=plan.p;assert p.D==256 and p.M%128==0
        root=Path(__file__).resolve().parent
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma256.cuh').read_text())
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_full_width_prefix_dx').kernel('mw_d256_full_width_prefix_dx')
        barrier_bytes=((16*slots+16+127)//128)*128
        self.smem=768*k_tile*slots+barrier_bytes;self.k.set_max_dynamic_smem(self.smem);self.grid=p.M//128
        L=T._launch_module()
        a=L.tensor_map(plan.dx.input,[64,k_tile,2],dims=[64,2304,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2=l2)
        w=L.tensor_map(plan.dx.weights,[64,k_tile,4],dims=[64,2304,4],strides_bytes=[512,128],swizzle='128B',l2=l2)
        y=L.tensor_map(p.tensors[10],[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,w,y])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2),(self.registers,'insufficient dynamic register pool')
    def __call__(self):self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
