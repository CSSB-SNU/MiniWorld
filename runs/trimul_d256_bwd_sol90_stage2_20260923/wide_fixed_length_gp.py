"""Specialize contraction address arithmetic and K loops for the live length."""
from wide_staged_epilogue_gp import StagedEpilogueGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FixedLengthGP:
    def __init__(self,plan,unroll=1):
        prior=plan.contract_gp if isinstance(plan.contract_gp,StagedEpilogueGP) else StagedEpilogueGP(plan)
        self.__dict__.update(prior.__dict__)
        body=prior.source_text.replace('p.N',str(self.p.n))
        marker=f' for(int ki=0,it=0;ki<{self.p.n};ki+=64,++it)'
        assert body.count(marker)==1
        body=body.replace(marker,f' #pragma unroll {unroll}\n'+marker)
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_fixed_length_gp')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_fixed_length_gp').kernel('mw_wide_fixed_length_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
