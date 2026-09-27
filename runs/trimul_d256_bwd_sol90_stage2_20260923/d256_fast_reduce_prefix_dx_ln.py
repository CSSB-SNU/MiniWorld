"""Parallel affine reduction and vectorized exact input-weight sums."""
from pathlib import Path
from d256_saved_stats_prefix_dx_ln import SavedStatsPrefixDxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FastReducePrefixDxLN(SavedStatsPrefixDxLN):
    def __init__(self,plan,emit_dxn=False,slots=4,k_tile=64,depth=0):
        super().__init__(plan,emit_dxn,slots,k_tile,depth)
        root=Path(__file__).resolve().parent;body=self.source_text
        begin=body.index('struct ReduceParams')
        body=body[:begin]+(root/'d256_prefix_dx_ln_fast_reduce.cuh').read_text()
        body=body.replace('mw_d256_saved_stats_prefix_dx_ln','mw_d256_fast_reduce_prefix_dx_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_fast_reduce_prefix_dx_ln')
        self.k=unit.kernel('mw_d256_fast_reduce_prefix_dx_ln');self.k.set_max_dynamic_smem(self.smem)
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        assert self.registers*384>=128*(32+224*2)
    def __call__(self):
        self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
        self.first.launch((8,self.chunks,1),(256,1,1),[self.rp],0)
        self.finish.launch((256,1,1),(128,1,1),[self.rp],0)
