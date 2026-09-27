"""Modern resident dP path computes saved-product derivatives before dNorm."""
from d256_scoped_slim_dn_ln import ScopedSlimDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
import torch

class ModernGateDnLN(ScopedSlimDnLN):
    def __init__(self,plan,emit_dn=False,n256=True):
        super().__init__(plan,emit_dn,n256)
        p=plan.p;body=self.source_text
        body=body.replace('CUtensorMap tri,dt,dp,wp;','CUtensorMap tri,dt,dp,wp,proj_in,gate_in,dy_in,dg_out;')
        body=body.replace('int M;','const bf* ds;int M,L;')
        body=body.replace('TMN_DEVI void producer(','''TMN_DEVI float saved_sigmoid(float g){float t;float x=__fmul_rn(.5f,g);asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(x));return __fmaf_rn(t,.5f,.5f);}
TMN_DEVI void put_product(const CUtensorMap* map,uint8_t* sm,int row){
 asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,0}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row):"memory");
}
TMN_DEVI void producer(''')
        marker='  mbar_arrive_expect_tx(bars,32768);\n  tma_load_3d(sm+BUF,&p.dp,bars,0,row,0);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  mbar_arrive_expect_tx(bars,98304);
  tma_load_3d(sm,&p.proj_in,bars,0,row,0);
  tma_load_3d(sm+32768,&p.gate_in,bars,0,row,0);
  tma_load_3d(sm+BUF,&p.dy_in,bars,0,row,0);''')
        marker='  uint32_t fa[16][4];load_frag_bf16<16,8192>(fa,smem_u32(sm+BUF),warp*16,lane);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  uint32_t fa[16][4],gate_grad[16][4];
  #pragma unroll
  for(int k=0;k<16;++k){
   #pragma unroll
   for(int j=0;j<4;++j){
    int r=warp*16+lane/4+8*(j&1),c=k*16+2*(lane%4)+8*(j/2);
    unsigned off=(c/64)*8192+swz128(r,(c%64)*2);
    uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+32768+off);
    uint32_t dy=*reinterpret_cast<uint32_t*>(sm+BUF+off),ds=reinterpret_cast<const uint32_t*>(p.ds)[(((row+r)%p.L)*256+c)/2];
    float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));
    float g0=saved_sigmoid(bf16lo(ga)),g1=saved_sigmoid(bf16hi(ga));
    fa[k][j]=pack_bf16(a*g0,b*g1);
    gate_grad[k][j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
    *reinterpret_cast<uint32_t*>(sm+BUF+off)=fa[k][j];
   }
  }
  named_bar_sync(1,128);
  #pragma unroll
  for(int k=0;k<16;++k){
   int mat=lane/8;
   uint32_t dst=smem_u32(sm+(k/4)*8192)+swz128((k%4)*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
   stsm_x4_t(dst,gate_grad[k][0],gate_grad[k][1],gate_grad[k][2],gate_grad[k][3]);
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){put_product(&p.dp,sm+BUF,row);put_tile(&p.dg_out,sm,row,0);tma_store_commit();tma_store_wait_read<0>();}
  named_bar_sync(1,128);''')
        body=body.replace('mw_d256_scoped_slim_dn_ln','mw_d256_modern_gate_dn_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=1','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_modern_gate_dn_ln').kernel('mw_d256_modern_gate_dn_ln');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=self.params.fields
        rowmap=lambda x:L.tensor_map(x,[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        # D256 Lt schedule exposes raw A/B/C tensors in its argument tuples.
        self.params=L.Struct(fields[:4]+[
            rowmap(plan.schedule.args['proj'][2]),rowmap(plan.schedule.args['gate'][2]),rowmap(p.dy),
            L.tensor_map(plan.prefix_gate.prefix,[64,64,4],dims=[p.M,64,4],strides_bytes=[p.M*2,p.M*128],swizzle='128B',l2='128B')
        ]+fields[4:-1]+[p.ds,p.M,p.n])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy;self.rows=self.grid
