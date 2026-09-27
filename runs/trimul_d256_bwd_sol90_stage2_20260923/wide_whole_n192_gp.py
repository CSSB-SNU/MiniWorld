"""M128N192 contraction with whole TMA tiles and cache-resident exact masks."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WholeN192GP:
    def __init__(self,plan,unroll=False):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        p=self.p;d=p.D;n=p.n;assert n==384
        root=Path(__file__).resolve().parent
        body=(root/'wide_n192_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma192_offset.cuh').read_text())
        body=body.replace('BAR=114688','BAR=98304')
        helper='''TMN_DEVI void load4(void* dst,const CUtensorMap* map,uint64_t* bar,int c2,int c3){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,%3,%4}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c2),"r"(c3):"memory");
}
'''
        body=body.replace('template<int MODE> TMN_DEVI void load_input(',helper+'template<int MODE> TMN_DEVI void load_input(')
        start=body.index(' for(int g=0;g<2;++g){int m=mi+64*g;')
        end=body.index('\n}\ntemplate<int MODE> TMN_DEVI void run',start)
        body=body[:start]+''' load4(sm+slot*INPUT,p.a+MODE,bar+slot,(TA?mi:ki)/64,(ch*p.N+(TA?ki:mi))/64);
 load4(sm+slot*INPUT+16384,p.b+MODE,bar+slot,(TB?ni:ki)/64,(ch*p.N+(TB?ki:ni))/64);'''+body[end:]
        start=body.index('  for(int wm=0;wm<2;++wm)for(int wn=0;wn<3;++wn)')
        end=body.index(' mbar_wait(bar+2,0);',start)
        body=body[:start]+'''  load4(sm,&p.premap,bar+2,ni/64,((rank*64+pc)*p.N+mi)/64);
  load4(sm+49152,&p.premap,bar+2,ni/64,((rank*64+pc+32)*p.N+mi)/64);
 }
'''+body[end:]
        start=body.index(' static_for<3>');end=body.index('\n}\nextern',start)
        body=body[:start]+''' static_for<48>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(WG*3+c/64)*8192+swz128(r,(c%64)*2);
  uint32_t raw=pack_bf16(v[j],v[j+1]),masked;
  uint32_t mask=__ldg(reinterpret_cast<const uint32_t*>(p.mask+size_t(mi+WG*64+r)*p.N+ni+c));
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+49152+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  *reinterpret_cast<uint32_t*>(sm+off)=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(sm+49152+off)=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
 });
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  for(int g=0;g<2;++g){
   asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0,{0,0,%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*49152)),"r"(ni/64),"r"((outch*p.N+mi)/64):"memory");
  }tma_store_commit();tma_store_wait_all();
 }
'''+body[end:]
        start=body.index(' int tiles=(p.N/128)*(p.N/192),mode=');end=body.index('\n int mi=',start)
        body=body[:start]+''' int tiles=(p.N/128)*(p.N/192),half=blockIdx.x/(2*D*tiles),rem=blockIdx.x%(2*D*tiles);
 int ch=rem/(2*tiles),mode=2*half+rem%2,tile=(rem/2)%tiles;'''+body[end:]
        if unroll:body=body.replace('for(int ki=0,it=0;ki<p.N;ki+=64,++it)', '#pragma unroll\n for(int ki=0,it=0;ki<384;ki+=64,++it)')
        body=body.replace('mw_wide_n192_contract_gp','mw_wide_whole_n192_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_whole_n192_gp').kernel('mw_wide_whole_n192_gp')
        self.smem=98304+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        def tm(x,channels,box):return L.tensor_map(x,box,dims=[64,64,n//64,channels*n//64],strides_bytes=[n*2,128,n*128],swizzle='128B',l2='128B')
        ab=p.front.ab;aa=(p.dt[:d],p.dt[:d],ab[3*d:],ab[d:2*d]);bb=(ab[2*d:3*d],ab[:d],p.dt[d:],p.dt[d:])
        fields=prior.params.fields.copy()
        for i,x in enumerate(aa):fields[i]=tm(x,d,(64,64,2,1) if i==1 else (64,64,1,2))
        for i,x in enumerate(bb):fields[4+i]=tm(x,d,(64,64,3,1) if i!=2 else (64,64,1,3))
        fields[16]=tm(plan.pre,8*d,(64,64,3,2))
        for i,x in enumerate(p.gp):fields[18+i]=tm(x,2*d,(64,64,3,2))
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)*(self.p.n//192),1,1),(256,1,1),[self.params],self.smem)
