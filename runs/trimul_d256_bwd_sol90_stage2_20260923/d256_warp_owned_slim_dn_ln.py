"""Warp-owned 64x32 transposes remove inter-warp exchanges from the slim LN."""
from pathlib import Path
from d256_scoped_slim_dn_ln import ScopedSlimDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
import torch
class WarpOwnedSlimDnLN(ScopedSlimDnLN):
    def __init__(self,plan,emit_dn=False,n256=True):
        super().__init__(plan,emit_dn,n256)
        body=self.source_text;root=Path(__file__).resolve().parent
        begin=body.index('template<bool INVERSE>');end=body.index('TMN_DEVI void put_tile',begin)
        body=body[:begin]+(root/'warp_owned_transpose32.cuh').read_text()+'\n'+body[end:]
        body=body.replace('mw_d256_scoped_slim_dn_ln','mw_d256_warp_owned_slim_dn_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=1','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_warp_owned_slim_dn_ln').kernel('mw_d256_warp_owned_slim_dn_ln');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.grid=torch.cuda.get_device_properties(plan.p.x.device).multi_processor_count*self.occupancy;self.rows=self.grid
