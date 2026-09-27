"""Bound gate-gradient registers to a single 64-channel output tile."""
from d256_modern_gate_dn_ln import ModernGateDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
import torch

class ChunkedGateDnLN(ModernGateDnLN):
    def __init__(self,plan,emit_dn=False,n256=True):
        super().__init__(plan,emit_dn,n256)
        body=self.source_text
        start=body.index('  uint32_t fa[16][4],gate_grad[16][4];')
        stop=body.index('  fence_proxy_async();named_bar_sync(1,128);',start)
        body=body[:start]+'''  #pragma unroll
  for(int kc=0;kc<4;++kc){
   uint32_t gate_grad[4][4];
   #pragma unroll
   for(int q=0;q<4;++q){
    #pragma unroll
    for(int j=0;j<4;++j){
     int r=warp*16+lane/4+8*(j&1),c=kc*64+q*16+2*(lane%4)+8*(j/2);
     unsigned off=kc*8192+swz128(r,(c%64)*2);
     uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+32768+off);
     uint32_t dy=*reinterpret_cast<uint32_t*>(sm+BUF+off),ds=reinterpret_cast<const uint32_t*>(p.ds)[(((row+r)%p.L)*256+c)/2];
     float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));
     float g0=saved_sigmoid(bf16lo(ga)),g1=saved_sigmoid(bf16hi(ga));
     *reinterpret_cast<uint32_t*>(sm+BUF+off)=pack_bf16(a*g0,b*g1);
     gate_grad[q][j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
    }
   }
   named_bar_sync(1,128);
   #pragma unroll
   for(int q=0;q<4;++q){
    int mat=lane/8;
    uint32_t dst=smem_u32(sm+kc*8192)+swz128(q*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
    stsm_x4_t(dst,gate_grad[q][0],gate_grad[q][1],gate_grad[q][2],gate_grad[q][3]);
   }
  }
  uint32_t fa[16][4];load_frag_bf16<16,8192>(fa,smem_u32(sm+BUF),warp*16,lane);
'''+body[stop:]
        body=body.replace('mw_d256_modern_gate_dn_ln','mw_d256_chunked_gate_dn_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=1','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_chunked_gate_dn_ln').kernel('mw_d256_chunked_gate_dn_ln');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.grid=torch.cuda.get_device_properties(plan.p.x.device).multi_processor_count*self.occupancy;self.rows=self.grid
