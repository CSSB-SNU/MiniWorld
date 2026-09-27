"""Full N384 contraction with staged saved preactivations and fused GP."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class FullSpatialGP:
    def __init__(self,plan,groups=2):
        p=plan.p;assert p.n==384 and groups in (1,2)
        self.p=p;root=Path(__file__).resolve().parent
        helper=(root/'mma256.cuh').read_text()
        helper=helper.replace('template<int TA,int TB>','template<int OA,int OB,int TA,int TB>').replace('void mma256(','void mma256_off(')
        helper=helper.replace('{.reg .pred p;setp.ne.b32 p,%130,0;', '{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%128,%133;add.u64 bx,%129,%134;setp.ne.b32 p,%130,0;')
        helper=helper.replace(',%128,%129,p,1,1,',',ax,bx,p,1,1,').replace('"n"(TB));','"n"(TB),"n"(OA>>4),"n"(OB>>4));')
        body=(root/'wide_full_spatial_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()+'\n'+helper)
        epi=[]
        for i,(width,offset) in enumerate(((256,0),(128,256))):
            epi.append(f'''static_for<{width//4}>([&](auto jj){{constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c={offset}+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(wg*6+c/64)*8192+swz128(r,(c%64)*2);
  uint32_t raw=pack_bf16(v{i}[j],v{i}[j+1]),masked,mask=*reinterpret_cast<const uint32_t*>(p.mask+(mi+wg*64+r)*N+c);
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+EPI+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  *reinterpret_cast<uint32_t*>(sm+off)=dp;*reinterpret_cast<uint32_t*>(sm+EPI+off)=dg;
 }});''')
        body=body.replace('// EPILOGUE','\n'.join(epi));self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DROW_GROUPS={groups}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_full_spatial_gp').kernel('mw_wide_full_spatial_gp')
        self.smem=(groups+6)*8192*3+128;self.threads=128*groups;self.grid=4*p.D*(384//(64*groups));self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=p.D;h=2*d;n=384
        def tm(t,trans,count,channels=d):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*channels,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*channels//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=L.Struct([*[tm(t,i==1,groups) for i,t in enumerate(aa)],*[tm(t,i!=2,6) for i,t in enumerate(bb)],tm(plan.pre,True,6,8*d),*[tm(t,True,6,h) for t in p.gp],plan.f.mask])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
