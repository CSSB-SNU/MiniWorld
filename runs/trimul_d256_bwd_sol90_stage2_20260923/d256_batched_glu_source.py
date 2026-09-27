"""Pack rounded preactivations early and group independent sigmoid instructions."""
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class BatchedGLUSource:
    def __init__(self,plan,variant=1):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);body=prior.source_text
        assert variant in (0,1,2,3)
        start=body.index('TMN_DEVI void packed_glu(');end=body.index('\n}',start)+2
        if variant==0:
            helper=body[start:end].replace('float (&a)[32]','uint32_t (&a)[16]')
            helper=helper.replace('pack_bf16(a[q*8+j*2],a[q*8+j*2+1])','a[q*4+j]').replace('pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1])','a[(q+2)*4+j]')
        else:
            qs=2 if variant==2 else 1
            helper=f'''TMN_DEVI void packed_glu(uint32_t (&a)[16],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){{
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int first=0;first<2;first+={qs}){{
  uint32_t incoming[{qs}][4],masked[{qs}][4],dg[{qs}][4],dp[{qs}][4];float g[{qs*8}];
  #pragma unroll
  for(int q=0;q<{qs};++q){{
   ldsm_x4_t(incoming[q],smem_u32(s+32768)+swz128((first+q)*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
   #pragma unroll
   for(int j=0;j<4;++j){{
    uint32_t m=(j&1)?mb:ma,gr=a[(first+q)*4+j];
    asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked[q][j]):"r"(incoming[q][j]),"r"(m));
    g[q*8+2*j]=math::ex2_approx_ftz(__fmul_rn(-1.4426950408889634f,bf16lo(gr)));
    g[q*8+2*j+1]=math::ex2_approx_ftz(__fmul_rn(-1.4426950408889634f,bf16hi(gr)));
   }}
  }}
  {'fence_regs(g);' if variant==3 else ''}
  #pragma unroll
  for(int j=0;j<{qs*8};++j)g[j]=math::rcp_approx_ftz(__fadd_rn(1.f,g[j]));
  {'fence_regs(g);' if variant==3 else ''}
  #pragma unroll
  for(int q=0;q<{qs};++q){{
   #pragma unroll
   for(int j=0;j<4;++j){{
    uint32_t pr=a[(first+q+2)*4+j];float ga=g[q*8+j*2],gb=g[q*8+j*2+1],pa=bf16lo(pr),pb=bf16hi(pr);
    dg[q][j]=pack_bf16(((bf16lo(masked[q][j])*pa)*ga)*(1.f-ga),((bf16hi(masked[q][j])*pb)*gb)*(1.f-gb));
    dp[q][j]=pack_bf16(bf16lo(masked[q][j])*ga,bf16hi(masked[q][j])*gb);
   }}
   uint32_t addr=swz128((first+q)*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
   stsm_x4_t(smem_u32(sg)+addr,dg[q][0],dg[q][1],dg[q][2],dg[q][3]);
   stsm_x4_t(smem_u32(sg+4096)+addr,dp[q][0],dp[q][1],dp[q][2],dp[q][3]);
  }}
 }}
}}'''
        body=body[:start]+helper+body[end:]
        marker='  packed_glu(pre,xn,sm+DERIV,ma,mb);';assert body.count(marker)==1
        body=body.replace(marker,'''  uint32_t prepack[16];static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;prepack[j]=pack_bf16(pre[2*j],pre[2*j+1]);});
  fence_regs(prepack);packed_glu(prepack,xn,sm+DERIV,ma,mb);''')
        body=body.replace('mw_d256_register_budget_source','mw_d256_batched_glu_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_batched_glu_source',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_batched_glu_source').kernel('mw_d256_batched_glu_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
