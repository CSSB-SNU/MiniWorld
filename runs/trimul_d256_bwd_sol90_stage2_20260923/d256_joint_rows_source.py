"""Paired source rows with half-column dW ownership, preserving K order."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class JointRowsSource:
    def __init__(self,plan,consumer_regs=224,splits=None):
        self.p=plan.p;self.splits=plan.b7.splits if splits is None else splits
        root=Path(__file__).resolve().parent
        body=(root/'d256_joint_rows_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',(root/'packed_glu.cuh').read_text())
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DCONSUMER_REGS={consumer_regs}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_joint_rows_source').kernel('mw_d256_joint_rows_source');self.smem=197248;self.k.set_max_dynamic_smem(self.smem)
        f=plan.b7.params.fields;self.params=T._launch_module().Struct([*f[:9],*f[13:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
    def __call__(self):self.k.launch((32*self.splits,1,1),(384,1,1),[self.params],self.smem)
