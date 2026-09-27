"""Overlap consecutive ordered K groups before reusing retired TMA slots."""
from d256_full_spatial_contract import FullSpatialContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class AsyncSpatialContract(FullSpatialContract):
    def __init__(self,plan,wait_depth=1,entry_sync=True):
        super().__init__(plan,2,3,256)
        body=self.source_text
        prefetch='  if(threadIdx.x==0 && step+SLOTS-1<6)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-1);'
        assert body.count(prefetch)==1;body=body.replace(prefetch,'')
        fence='fence_regs(v0);fence_regs(v1);'
        assert body.count(fence)==2
        body=body.replace(fence+'wgmma_fence();','')
        body=body.replace('float v0[128]={},v1[64]={};','float v0[128]={},v1[64]={};'+fence+'wgmma_fence();')
        old='});wgmma_commit();wgmma_wait<0>();'+fence+'\n }\n __syncthreads();'
        assert body.count(old)==1
        body=body.replace(old,f'''}});wgmma_commit();wgmma_wait<{wait_depth}>();
  // All row owners retire the slot that the producer is about to recycle.
  __syncthreads();
{prefetch}
 }}
 wgmma_wait<0>();{fence}__syncthreads();''')
        if not entry_sync:
            body=body.replace('mbar_wait(bar+slot,(step/SLOTS)&1);__syncthreads();','mbar_wait(bar+slot,(step/SLOTS)&1);')
        body=body.replace('mw_d256_full_spatial_contract','mw_d256_async_spatial_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DROW_GROUPS=2','-DGRID_ORDER=1','-DMIN_BLOCKS=1','-DCONTRACT_SLOTS=3']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_async_spatial_contract').kernel('mw_d256_async_spatial_contract')
        self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
