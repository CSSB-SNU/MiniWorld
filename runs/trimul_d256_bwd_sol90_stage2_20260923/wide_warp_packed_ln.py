"""Packed row16 LN: each warp synchronizes only its own transpose tile."""
import torch
from wide_packed_retained_ln import PackedRetainedLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class WarpPackedLN(PackedRetainedLN):
    def __init__(self,p,original):
        self.__dict__.update(original.__dict__);assert self.rows==16 and p.D in (384,512)
        body=original.source_text
        begin=body.index('template<bool INVERSE>');end=body.index('TMN_DEVI void put_tile',begin)
        helper=body[begin:end];assert helper.count('named_bar_sync(1,128);')==2
        helper=helper.replace('named_bar_sync(1,128);','__syncwarp();',1)
        body=body[:begin]+helper+body[end:]
        body=body.replace('mw_packed_retained_ln','mw_warp_packed_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}','-DLN_ROWS=16','-DLN_DN_TMA=1','-DLN_FENCE=0','-DLN_MINBLOCKS=2']
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_warp_packed_ln').kernel('mw_warp_packed_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
