"""Match the initial CTA pool to smaller dynamically allocated consumer budgets."""
from d256_register_layout_source import RegisterLayoutSource
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class ReadWaitRegisterBudgetSource(RegisterLayoutSource):
    def __init__(self,plan,budget=192):
        super().__init__(plan,True,False)
        assert budget in (160,168,176,184,192,200,208,216)
        body=self.source_text.replace('setmaxnreg_inc<224>()',f'setmaxnreg_inc<{budget}>()')
        body=body.replace('tma_store_wait_all();','tma_store_wait_read<0>();')
        end=body.rfind('extern \"C\" __global__');point=body.rfind('}',0,end)
        body=body[:point]+' if(tid==0)tma_store_wait_all();\n'+body[point:]
        body=body.replace('mw_d256_register_layout_source','mw_d256_readwait_register_budget_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);initial=(32+budget)//2
        self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_readwait_register_budget_source',initial,budget)
        self.k=T.load_unit(str(self.cubin),'mw_d256_readwait_register_budget_source').kernel('mw_d256_readwait_register_budget_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        assert self.registers==initial and self.occupancy==2
