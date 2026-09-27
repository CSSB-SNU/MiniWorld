"""Use one bulk TMA transaction for each aligned compact row interior."""
from wide_tiled_compact_mask_gp import TiledCompactMaskGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class BulkCompactMaskGP(TiledCompactMaskGP):
    def __init__(self,plan,mapping):
        super().__init__(plan,mapping)
        body=self.source_text
        start=body.index('  for(int i=threadIdx.x;i<128*17;i+=256){')
        end=body.index('  __syncthreads();',start)
        body=body[:start]+'''  if(threadIdx.x<128){
   int r=threadIdx.x,row=(mi+r)*p.N+ni,begin=p.rows[row],end=p.rows[row+128];
   int head=begin&7,tail=head+end-begin,first=(head+7)&~7,last=tail&~7;
   auto dst=reinterpret_cast<uint16_t*>(p.gp[2*SIDE+g])+size_t(outch)*p.N*p.N+(begin&~7);
   if(last>first){
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"::"l"(dst+first),"r"(smem_u32(staging+r*136+first)),"r"((last-first)*2):"memory");
   }
   for(int q=head;q<tail && q<first;++q)dst[q]=staging[r*136+q];
   for(int q=max(last,first);q<tail;++q)dst[q]=staging[r*136+q];
   tma_store_commit();tma_store_wait_all();
  }
'''+body[end:]
        # The plain bulk engine must observe the compaction stores before use.
        marker='  __syncthreads();\n  if(threadIdx.x<128)'
        assert body.count(marker)==1
        body=body.replace(marker,'  fence_proxy_async();__syncthreads();\n  if(threadIdx.x<128)')
        body=body.replace('mw_wide_tiled_compact_mask_gp','mw_wide_bulk_compact_mask_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_bulk_compact_mask_gp').kernel('mw_wide_bulk_compact_mask_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
