from pathlib import Path
import os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class CompactStreamContractGP:
    def __init__(self,plan,grid=264,minblocks=2):
        self.__dict__.update(plan.contract_gp.__dict__)
        root=Path(__file__).resolve().parent
        body=(root/'wide_compact_stream_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root.parent/'trimul_d256_bwd_sol90_20260923/mma.cuh').read_text())
        if os.environ.get('GP_STATIC_N','0')=='1':
            body=body.replace('p.N',str(self.p.n))
            body=body.replace('  for(int it=0;it<steps;++it)', '  #pragma unroll 1\n  for(int it=0;it<steps;++it)')
        self.source_text=body;self.grid=grid;self.smem=114688+128
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}',f'-DSTREAM_MINBLOCKS={minblocks}','-DSTORE_READ_CREDIT=1']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_compact_stream_contract_gp').kernel('mw_wide_compact_stream_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fun=drv.d.CUfunction(int(self.k.handle))
        attr=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fun)))
        self.registers=attr('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=attr('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fun,512,self.smem)))
    def __call__(self):self.k.launch((self.grid,1,1),(512,1,1),[self.params],self.smem)
