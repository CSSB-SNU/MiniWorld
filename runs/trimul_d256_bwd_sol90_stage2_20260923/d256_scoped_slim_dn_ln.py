"""Retire affine accumulators between row tiles so GEMM can use a wider RS tile."""
from d256_modern_slim_dn_ln import ModernSlimDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
import torch
class ScopedSlimDnLN(ModernSlimDnLN):
    def __init__(self,plan,emit_dn=False,n256=False):
        super().__init__(plan,emit_dn,True);body=self.source_text
        begin=body.index(' float gg[16]={},bb[16]={};');end=body.index(' for(int row=',begin)
        body=body[:begin]+' float* saved_affine=reinterpret_cast<float*>(sm+BARS+640);\n'+body[end:]
        marker=' for(int c=tid;c<H;c+=256)reinterpret_cast<float*>(sm+BARS+640)[c]=p.gamma[c];'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<4096;c+=256)reinterpret_cast<float*>(sm+BARS+640)[c]=0;')
        marker='  if constexpr(DNS_STATS_TMA)mbar_wait(bars+6+2*NBUF,round&1);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  float gg[16],bb[16],gamma[16];
  #pragma unroll
  for(int q=0;q<16;++q){int c=lane+q*32;gg[q]=saved_affine[warp*H+c];bb[q]=saved_affine[(4+warp)*H+c];gamma[q]=p.gamma[c];}
'''+marker)
        body=body.replace('DNS_CACHE_GAMMA?gamma[q]:reinterpret_cast<float*>(sm+BARS+640)[c]','gamma[q]')
        marker='  }\n }\n if(tid==0)tma_store_wait_all();'
        assert body.count(marker)==1
        body=body.replace(marker,'''  }
  #pragma unroll
  for(int q=0;q<16;++q){int c=lane+q*32;saved_affine[warp*H+c]=gg[q];saved_affine[(4+warp)*H+c]=bb[q];}
 }
 if(tid==0)tma_store_wait_all();''')
        begin=body.index(' float* parts=reinterpret_cast<float*>(sm);');end=body.index(' for(int c=tid;c<H;c+=128)',begin)
        body=body[:begin]+' float* parts=saved_affine;\n'+body[end:]
        fields=self.params.fields.copy()
        if n256:
            body=body.replace('(step%4)*64,(step/4)*2','(step%8)*32,(step/8)*4')
            body=body.replace('for(int col=0;col<H;col+=128)','for(int col=0;col<H;col+=256)')
            body=body.replace('float v[64]={};','float v0[64]={},v1[64]={};').replace('fence_regs(v);','fence_regs(v0);fence_regs(v1);')
            body=body.replace('static_for<4>([&](auto kk)','static_for<8>([&](auto kk)').replace('static_for<4>([&](auto qq)','static_for<2>([&](auto qq)')
            marker='uint64_t b0=smem_desc(smem_u32(sm+BUF+slot*16384),8192,1024,1);'
            assert body.count(marker)==1
            body=body.replace(marker,'uint64_t b0=smem_desc(smem_u32(sm+BUF+slot*16384),4096,1024,1);uint64_t b1=smem_desc(smem_u32(sm+BUF+slot*16384+8192),4096,1024,1);')
            marker='mma_rs128<q*2048>(v,fa[k*4+q],uint32_t(b0),uint32_t(b0>>32),k>0||q>0);'
            assert body.count(marker)==1
            body=body.replace(marker,marker.replace('(v,','(v0,').replace('k*4+q','k*2+q')+marker.replace('(v,','(v1,').replace('k*4+q','k*2+q').replace('b0','b1'))
            marker='''    unsigned raw=pack_bf16(v[j],v[j+1]);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+dnpos(r,col+c))=raw;
    if constexpr(EMIT_DN)*reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+col+c)=raw;'''
            assert body.count(marker)==1
            body=body.replace(marker,'''
    unsigned raw0=pack_bf16(v0[j],v0[j+1]),raw1=pack_bf16(v1[j],v1[j+1]);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+dnpos(r,col+c))=raw0;
    *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+dnpos(r,col+128+c))=raw1;
    if constexpr(EMIT_DN){
     *reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+col+c)=raw0;
     *reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+col+128+c)=raw1;
    }''')
            fields[3]=T._launch_module().tensor_map(plan.leaves[6],[64,32,4],dims=[64,256,8],strides_bytes=[1024,128],swizzle='128B',l2='128B')
        body=body.replace('mw_d256_modern_slim_dn_ln','mw_d256_scoped_slim_dn_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=1','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_scoped_slim_dn_ln').kernel('mw_d256_scoped_slim_dn_ln');self.smem=98304+640+16384;self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.grid=torch.cuda.get_device_properties(plan.p.x.device).multi_processor_count*self.occupancy;self.rows=self.grid
