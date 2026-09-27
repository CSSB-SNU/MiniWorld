"""Preserve shared tiles while grouping their transfers in four-dimensional maps."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class WholeTmaGP:
    def __init__(self,plan,inputs=False):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        p=plan.p;n=p.n;d=p.D;h=2*d
        assert n==384
        body=prior.source_text
        marker='template<int MODE,int PLANE> TMN_DEVI void load_epi('
        helper='''TMN_DEVI void load4(void* dst,const CUtensorMap* map,uint64_t* bar,int c2,int c3){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,%3,%4}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c2),"r"(c3):"memory");
}
'''
        assert body.count(marker)==1
        body=body.replace(marker,helper+marker)
        old=''' for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
  int q=(wm*2+wn)*8192;
  if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
  else tma_load_2d(sm+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);
 }
'''
        new=''' if constexpr(PLANE<2)load4(sm+PLANE*32768,&p.premap,bar+6,ni/64,((rank*64+pc+PLANE*32)*p.N+mi)/64);
 else load4(sm+65536,&p.maskmap,bar+6,ni/64,mi/64);
'''
        assert body.count(old)==1
        body=body.replace(old,new)
        start=body.index('  for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){')
        end=body.index('  tma_store_commit();',start)
        body=body[:start]+'''  for(int g=0;g<2;++g){
   asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0,{0,0,%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*32768)),"r"(ni/64),"r"((outch*p.N+mi)/64):"memory");
  }
'''+body[end:]
        L=T._launch_module()
        def tm(x,channels,box=(64,64,2,2)):
            return L.tensor_map(x,box,dims=[64,64,n//64,channels*n//64],strides_bytes=[n*2,128,n*128],swizzle='128B',l2='128B')
        fields=prior.params.fields.copy()
        # Four A, four B, eight pointers, then pre/mask and four output maps.
        assert len(fields)==23,len(fields)
        fields[16]=tm(plan.pre,8*d);fields[17]=tm(plan.f.mask,1)
        for i,x in enumerate(p.gp):fields[18+i]=tm(x,h)
        if inputs:
            old=''' for(int g=0;g<2;++g){int m=mi+64*g,n=ni+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
  tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
 }
'''
            new=''' load4(sm+slot*INPUT,p.a+MODE,bar+slot,(TA?mi:ki)/64,(ch*p.N+(TA?ki:mi))/64);
 load4(sm+slot*INPUT+16384,p.b+MODE,bar+slot,(TB?ni:ki)/64,(ch*p.N+(TB?ki:ni))/64);
'''
            assert body.count(old)==1
            body=body.replace(old,new)
            # load_input precedes load4 in the source.
            body=body.replace(helper,'')
            body=body.replace('template<int MODE> TMN_DEVI void load_input(',helper+'template<int MODE> TMN_DEVI void load_input(')
            ab=p.front.ab
            aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
            for i,x in enumerate(aa):fields[i]=tm(x,d,(64,64,2,1) if i==1 else (64,64,1,2))
            for i,x in enumerate(bb):fields[4+i]=tm(x,d,(64,64,2,1) if i!=2 else (64,64,1,2))
        self.params=L.Struct(fields)
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_whole_tma_gp')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_whole_tma_gp').kernel('mw_wide_whole_tma_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
