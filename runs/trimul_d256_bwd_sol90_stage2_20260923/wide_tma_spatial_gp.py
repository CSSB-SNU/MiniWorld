"""One row warpgroup with whole-map mask transfers and N192/N384 columns."""
from pathlib import Path
from wide_full_spatial_gp import FullSpatialGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class TmaSpatialGP(FullSpatialGP):
    def __init__(self,plan,width=192,slots=3):
        super().__init__(plan,1)
        assert width in (64,128,192,384) and slots in (2,3)
        p=plan.p;root=Path(__file__).resolve().parent;slabs=width//64;epi=64*width*2
        body=self.source_text
        body=body.replace('INPUT=(ROW_GROUPS+6)*8192,SLOTS=3,BAR=INPUT*SLOTS,EPI=ROW_GROUPS*49152',f'INPUT={(1+slabs)*8192},SLOTS={slots},EPI={epi},BAR=(INPUT*SLOTS>3*EPI?INPUT*SLOTS:3*EPI)')
        body=body.replace('pre,out[4];const bf* mask;', 'pre,out[4],maskmap;const bf* mask;')
        # Every CTA covers one independent spatial column range.
        body=body.replace('int ch,int mi,int step)', 'int ch,int mi,int ni,int step)')
        body=body.replace('int ch,int mi){', 'int ch,int mi,int ni){')
        body=body.replace('p,sm,bar,ch,mi,','p,sm,bar,ch,mi,ni,')
        body=body.replace('p,sm,bar,ch,mi);','p,sm,bar,ch,mi,ni);')
        body=body.replace('TB?0:ch*N/64','TB?ni/64:(ch*N+ni)/64')
        body=body.replace('(rank*64+pc+PLANE*32)*N+mi+wm*64,0)', '(rank*64+pc+PLANE*32)*N+mi+wm*64,ni/64)')
        body=body.replace('sm+PLANE*EPI+wm*49152', 'sm+PLANE*EPI+wm*EPI')
        body=body.replace('int slot=step%3','int slot=step%SLOTS').replace('(step/3)&1','(step/SLOTS)&1')
        body=body.replace('load<MODE>(p,sm,bar,ch,mi,ni,1);','if constexpr(SLOTS==3)load<MODE>(p,sm,bar,ch,mi,ni,1);')
        begin=body.index('  if(threadIdx.x==0){\n   if(step+2<6)');end=body.index('\n  fence_regs(v0)',begin)
        body=body[:begin]+'''  if(threadIdx.x==0){
   if(step+SLOTS-1<6)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-1);
   if constexpr(SLOTS==3){
    if(step==4)load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
    if(step==5)load_epi<MODE,1>(p,sm,bar,ch,mi,ni);
   }else if(step==5)load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
  }'''+body[end:]
        body=body.replace('if constexpr(ROW_GROUPS==2)if(threadIdx.x==0)load_epi', 'if constexpr(SLOTS==2)if(threadIdx.x==0)load_epi')
        begin=body.index(' mbar_wait(bar+3,0);__syncthreads();');end=body.index(' fence_proxy_async();__syncthreads();',begin)
        code=''' if(threadIdx.x==0){mbar_arrive_expect_tx(bar+4,EPI);tma_load_3d(sm+2*EPI,&p.maskmap,bar+4,0,mi,ni/64);}
 mbar_wait(bar+3,0);mbar_wait(bar+4,0);__syncthreads();
'''
        parts=((width,0),) if width!=384 else ((256,0),(128,256))
        for i,(columns,offset) in enumerate(parts):
            code+=f''' static_for<{columns//4}>([&](auto jj){{constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c={offset}+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(c/64)*8192+swz128(r,(c%64)*2);
  uint32_t raw=pack_bf16(v{i}[j],v{i}[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+2*EPI+off);
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+EPI+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  *reinterpret_cast<uint32_t*>(sm+off)=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(sm+EPI+off)=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
 }});
'''
        body=body[:begin]+code+body[end:]
        body=body.replace('[%0,{0,%2,0}]','[%0,{0,%2,%3}]').replace('"r"((ch+HALF*D)*N+mi+wm*64):','"r"((ch+HALF*D)*N+mi+wm*64),"r"(ni/64):')
        body=body.replace('for(int i=0;i<4;++i)mbar_init','for(int i=0;i<5;++i)mbar_init')
        body=body.replace('TILES=N/(64*ROW_GROUPS)',f'TILES=(N/64)*(N/{width})')
        body=body.replace('mi=((rem/2)%TILES)*(64*ROW_GROUPS);',f'mi=(((rem/2)%TILES)/(N/{width}))*64,ni=(((rem/2)%TILES)%(N/{width}))*{width};')
        if width!=384:
            if width==192:body=body.replace('constexpr int D=WIDTH,',(root/'mma192_offset.cuh').read_text()+'\nconstexpr int D=WIDTH,')
            body=body.replace('float v0[128]={},v1[64]={};',f'float v0[{width//2}]={{}};').replace('fence_regs(v1);','')
            body='\n'.join(line for line in body.splitlines() if 'mma128_off<k*' not in line)
            body=body.replace('mma256_off<k*',f'mma{width}_off<k*')
        body=body.replace('mw_wide_full_spatial_gp','mw_wide_tma_spatial_gp')
        resident={(64,2):6,(64,3):4,(128,2):4,(128,3):3,(192,2):3,(192,3):2,(384,2):1,(384,3):1}[width,slots]
        body=body.replace('__launch_bounds__(NT,1)',f'__launch_bounds__(NT,{resident})')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DROW_GROUPS=1']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_tma_spatial_gp').kernel('mw_wide_tma_spatial_gp')
        self.smem=max((1+slabs)*8192*slots,3*epi)+128;self.grid=4*p.D*6*(384//width);self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=p.D;h=2*d;n=384
        def tm(t,trans,count,channels=d):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*channels,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*channels//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=L.Struct([*[tm(t,i==1,1) for i,t in enumerate(aa)],*[tm(t,i!=2,slabs) for i,t in enumerate(bb)],tm(plan.pre,True,slabs,8*d),*[tm(t,True,slabs,h) for t in p.gp],tm(plan.f.mask,True,slabs,1),plan.f.mask])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
