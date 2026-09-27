"""Keep final warp reductions disjoint from the consumed prefix sums."""
import torch
from wide_ordered_pair_ln import OrderedPairLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SafeOrderedPairLN(OrderedPairLN):
    def __init__(self,p,rows=16,minblocks=3,narrow=False):
        super().__init__(p,rows,minblocks,narrow)
        body=self.source_text
        old='if(lane==0){sums[pair*64]=s0;sums[pair*64+1]=s1;}'
        assert body.count(old)==1
        body=body.replace(old,'if(lane==0){sums[128+pair*2]=s0;sums[128+pair*2+1]=s1;}')
        old='s0=sums[pair*64];s1=sums[pair*64+1];'
        assert body.count(old)==1
        body=body.replace(old,'s0=sums[128+pair*2];s1=sums[128+pair*2+1];')
        body=body.replace('mw_ordered_pair_ln','mw_safe_ordered_pair_ln')
        self.source_text=body;self.smem+=128
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_safe_ordered_pair_ln').kernel('mw_safe_ordered_pair_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
