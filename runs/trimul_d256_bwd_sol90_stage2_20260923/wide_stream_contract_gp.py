from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class StreamContractGP:
    def __init__(self,plan,grid=132,groups=2,read_credit=False):
        self.__dict__.update(plan.contract_gp.__dict__)
        root=Path(__file__).resolve().parent
        body=(root/'wide_stream_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        self.source_text=body;self.grid=grid;self.smem=229376+128;self.threads=(3+groups)*128
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}',f'-DGP_GROUPS={groups}',f'-DSTORE_READ_CREDIT={int(read_credit)}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_stream_contract_gp').kernel('mw_wide_stream_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fun=drv.d.CUfunction(int(self.k.handle))
        attr=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fun)))
        self.registers=attr('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=attr('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
