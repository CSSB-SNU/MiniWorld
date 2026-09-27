"""The TMA warpgroup also computes half of GLU using retired derivative storage."""
from two_role_register_pool import initial_pool_roles
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class TeamGLUSource:
    def __init__(self,plan,producer=64):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);consumer=256-producer
        body=prior.source_text
        helper='''
TMN_DEVI void team_glu(const Params& p,uint8_t* xn,uint8_t* sm,int slot){
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,q=threadIdx.x/128,mat=lane/8;
 int ra=w*16+lane/4;auto mask=reinterpret_cast<bf*>(sm+114816+slot*128);
 uint32_t ma=uint32_t(__bfloat16_as_ushort(mask[ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(mask[ra+8]))*0x10001u;
 uint32_t incoming[4],gr[4],pr[4],dg[4],dp[4];
 ldsm_x4_t(incoming,smem_u32(xn+32768)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
 #pragma unroll
 for(int j=0;j<4;++j){
  int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);
  gr[j]=*reinterpret_cast<uint32_t*>(sm+DERIV+swz128(r,c*2));
  pr[j]=*reinterpret_cast<uint32_t*>(sm+DERIV+swz128(r,(c+32)*2));
 }
 // Every preactivation has reached a register before any GP overwrites it.
 named_bar_sync(7,256);
 #pragma unroll
 for(int j=0;j<4;++j){
  uint32_t masked,m=(j&1)?mb:ma;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(incoming[j]),"r"(m));
  float ga=math::sigmoid(bf16lo(gr[j])),gb=math::sigmoid(bf16hi(gr[j])),pa=bf16lo(pr[j]),pb=bf16hi(pr[j]);
  dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
 }
 uint32_t off=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
 stsm_x4_t(smem_u32(sm+DERIV)+off,dg[0],dg[1],dg[2],dg[3]);
 stsm_x4_t(smem_u32(sm+DERIV+4096)+off,dp[0],dp[1],dp[2],dp[3]);
 named_bar_sync(7,256);fence_proxy_async();named_bar_sync(7,256);
}
'''
        marker='TMN_DEVI void compute_source';assert body.count(marker)==1
        body=body.replace(marker,helper+marker)
        start=body.index('  int ra=warp*16+lane/4;',body.index('TMN_DEVI void compute_source'))
        end=body.index('  if(tid==0){store_gp(',start)
        body=body[:start]+'''  static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   *reinterpret_cast<uint32_t*>(sm+DERIV+swz128(r,c*2))=pack_bf16(pre[j],pre[j+1]);
  });
  named_bar_sync(7,256);team_glu(p,xn,sm,slot);
'''+body[end:]
        start=body.index(' if(threadIdx.x<128){',body.index('extern "C" __global__'))
        end=body.rindex('\n}')
        body=body[:start]+f''' if(threadIdx.x<128){{
  setmaxnreg_dec<{producer}>();
  int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
  if(threadIdx.x==0){{
   mbar_arrive_expect_tx(bar+2,CH);
   for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
   load_input(p,sm,bar,begin*64,rank,0);
   if(begin+1<end)load_input(p,sm,bar,(begin+1)*64,rank,1);
  }}
  for(int tile=begin,it=0;tile<end;++tile,++it){{
   int slot=it%2;named_bar_sync(7,256);team_glu(p,sm+slot*INPUT,sm,slot);
   if(threadIdx.x==0 && tile+2<end){{mbar_wait(bar+3+slot,(it/2)&1);load_input(p,sm,bar,(tile+2)*64,rank,slot);}}
  }}
 }}else{{setmaxnreg_inc<{consumer}>();compute_source(p);}}
'''+body[end:]
        body=body.replace('__launch_bounds__(256,1)',f'__maxnreg__({consumer})').replace('mw_d256_register_budget_source','mw_d256_team_glu_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_roles(cubin,'mw_d256_team_glu_source',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_team_glu_source').kernel('mw_d256_team_glu_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
