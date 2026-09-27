"""Resident-register dP with current whole-map transfers and shared gamma."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class ModernSlimDnLN:
    def __init__(self,plan,emit_dn=False,n128=True):
        p=plan.p;assert p.D==256;self.p=p;root=Path(__file__).resolve().parent
        helper=(root/'dn_ln.cu').read_text().split('template <int OFF_BYTES>')[1].split('TMN_DEVI void dnorm')[0]
        body=(root/'dn_slim.cu').read_text().replace('// RS_MMA_HELPER','template <int OFF_BYTES>'+helper)
        body=body.replace('const bf* dn;','bf* dn;')
        marker='  for(int k=0;k<4;++k)tma_load_2d(sm+BUF+k*8192,&p.dp,bars,k*64,row);'
        assert body.count(marker)==1;body=body.replace(marker,'  tma_load_3d(sm+BUF,&p.dp,bars,0,row,0);')
        marker='   for(int g=0;g<2;++g)tma_load_2d(sm+BUF+slot*16384+g*8192,&p.wp,bars+2+slot,(step/4)*128+g*64,(step%4)*64);'
        assert body.count(marker)==1;body=body.replace(marker,'   tma_load_3d(sm+BUF+slot*16384,&p.wp,bars+2+slot,0,(step%4)*64,(step/4)*2);')
        marker='   for(int c=0;c<H;c+=64)tma_load_2d(sm+BUF+slot*LB+(c/64)*(ROWS*128),&p.tri,bars+6+slot,row+half*ROWS,c);'
        assert body.count(marker)==1;body=body.replace(marker,'   tma_load_3d(sm+BUF+slot*LB,&p.tri,bars+6+slot,row+half*ROWS,0,0);')
        marker='for(int c=0;c<H;c+=64)put_tile(&p.dt,tri+(c/64)*(ROWS*128),row+half*ROWS,c);'
        assert body.count(marker)==1;body=body.replace(marker,'put_tile(&p.dt,tri,row+half*ROWS,0);')
        body=body.replace('tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}]','tensor.3d.global.shared::cta.bulk_group [%0,{%2,%3,0}]')
        body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+BARS+640)[c]')
        marker=' __syncthreads();cooperative_groups::this_grid().sync();'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<H;c+=256)reinterpret_cast<float*>(sm+BARS+640)[c]=p.gamma[c];\n'+marker)
        if n128:
            outputs=','.join('%'+str(i) for i in range(64))
            constraints=','.join('"+f"(d['+str(i)+'])' for i in range(64))
            helper='''
template<int OFF_BYTES> TMN_DEVI void mma_rs128(float (&d)[64],const uint32_t (&a)[4],uint32_t lo0,uint32_t hi,int ac){
 asm volatile("{.reg .pred p;.reg .b32 lo;.reg .b64 b;setp.ne.b32 p,%70,0;add.u32 lo,%68,%71;mov.b64 b,{lo,%69};wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {OUTS},{%64,%65,%66,%67},b,p,1,1,1;}" : CONS : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(lo0),"r"(hi),"r"(ac),"n"(OFF_BYTES>>4));
}
'''.replace('OUTS',outputs).replace('CONS',constraints)
            body=body.replace('TMN_DEVI void producer(',helper+'\nTMN_DEVI void producer(')
            body=body.replace('float v0[32]={},v1[32]={};','float v[64]={};').replace('fence_regs(v0);fence_regs(v1);','fence_regs(v);')
            body=body.replace('uint64_t b0=smem_desc(smem_u32(sm+BUF+slot*16384),16,1024,1);','uint64_t b0=smem_desc(smem_u32(sm+BUF+slot*16384),8192,1024,1);')
            body=body.replace('    uint64_t b1=smem_desc(smem_u32(sm+BUF+slot*16384+8192),16,1024,1);','')
            body=body.replace('mma_rs_trans<q*2048>(v0,fa[k*4+q],uint32_t(b0),uint32_t(b0>>32),k>0||q>0);','mma_rs128<q*2048>(v,fa[k*4+q],uint32_t(b0),uint32_t(b0>>32),k>0||q>0);')
            body=body.replace('     mma_rs_trans<q*2048>(v1,fa[k*4+q],uint32_t(b1),uint32_t(b1>>32),k>0||q>0);','')
            begin=body.index('   static_for<32>([&](auto jj)');end=body.index('\n   named_bar_sync(1,128);',begin)
            body=body[:begin]+'''
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
    unsigned raw=pack_bf16(v[j],v[j+1]);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+dnpos(r,col+c))=raw;
    if constexpr(EMIT_DN)*reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+col+c)=raw;
   });'''+body[end:]
        else:
            a='    reinterpret_cast<bf*>(sm)[dnpos(r,col+64+c)]=__float2bfloat16_rn(v1[j]);'
            assert body.count(a)==1
            body=body.replace(a,a+'\n    if constexpr(EMIT_DN){p.dn[size_t(row+r)*H+col+c]=__float2bfloat16_rn(v0[j]);p.dn[size_t(row+r)*H+col+64+c]=__float2bfloat16_rn(v1[j]);}')
        body=body.replace('mw_d256_dn_slim','mw_d256_modern_slim_dn_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=1','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_modern_slim_dn_ln').kernel('mw_d256_modern_slim_dn_ln');self.smem=98304+640+2048;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        tri=lambda t:L.tensor_map(t,[32,64,8],dims=[p.M,64,8],strides_bytes=[p.M*2,p.M*128],swizzle='64B',l2='128B')
        dp=L.tensor_map(p.tensors[7],[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        wp=L.tensor_map(plan.leaves[6],[64,64,2],dims=[64,256,8],strides_bytes=[1024,128],swizzle='128B',l2='128B')
        self.params=L.Struct([tri(p.tri),tri(p.dt),dp,wp,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy;self.rows=self.grid
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
