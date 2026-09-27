"""Use TMA for contraction epilogue preactivation, mask, and GP traffic."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class TmaContractGP:
    def __init__(self,plan,fused=True,transposed=True):
        assert fused and transposed
        p=plan.p;d=p.D;h=2*d;n=p.n;self.p=p
        root=Path(__file__).resolve().parent
        body=(root/'wide_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        body=body.replace('BAR=2*INPUT','BAR=2*INPUT+24576')
        body=body.replace('bf* dr;int N;','bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;')
        point=' if(tid==0)load<MODE>(p,sm,bar,ch,mi,ni,0,0);'
        assert body.count(point)==1
        body=body.replace(point,''' if(tid==0){
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
  int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
  mbar_arrive_expect_tx(bar+2,24576);
  tma_load_2d(sm+32768,&p.premap,bar+2,ni,(rank*64+pc)*p.N+mi);
  tma_load_2d(sm+40960,&p.premap,bar+2,ni,(rank*64+pc+32)*p.N+mi);
  tma_load_2d(sm+49152,&p.maskmap,bar+2,ni,mi);
  load<MODE>(p,sm,bar,ch,mi,ni,0,0);
 }''')
        point=' constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);\n int outch='
        assert body.count(point)==1
        body=body.replace(point,' mbar_wait(bar+2,0);__syncthreads();\n'+point)
        point='uint32_t mask=pack_bf16(__bfloat162float(p.mask[row]),__bfloat162float(p.mask[row+1])),masked;'
        assert body.count(point)==1
        body=body.replace(point,'uint32_t smoff=swz128(r-mi,(c-ni)*2);uint32_t mask=*reinterpret_cast<uint32_t*>(sm+49152+smoff),masked;')
        begin=body.index('   float ga,gb,pa,pb;');end=body.index('   uint32_t dp=',begin)
        body=body[:begin]+'''   uint32_t gr=*reinterpret_cast<uint32_t*>(sm+32768+smoff),pr=*reinterpret_cast<uint32_t*>(sm+40960+smoff);
   float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
'''+body[end:]
        body=body.replace('*reinterpret_cast<uint32_t*>(p.gp[2*SIDE]+size_t(outch)*M+row)=dp;','*reinterpret_cast<uint32_t*>(sm+smoff)=dp;')
        body=body.replace('*reinterpret_cast<uint32_t*>(p.gp[2*SIDE+1]+size_t(outch)*M+row)=dg;','*reinterpret_cast<uint32_t*>(sm+8192+smoff)=dg;')
        point=' });\n}\nextern'
        assert body.count(point)==1
        body=body.replace(point,''' });
 fence_proxy_async();__syncthreads();
 if(tid==0){
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE),"r"(smem_u32(sm)),"r"(ni),"r"(outch*p.N+mi):"memory");
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+1),"r"(smem_u32(sm+8192)),"r"(ni),"r"(outch*p.N+mi):"memory");
  tma_store_commit();tma_store_wait_all();
 }
}
extern''')
        body=body.replace('mbar_init(bar+1,1);fence_barrier_init();','mbar_init(bar+1,1);mbar_init(bar+2,1);fence_barrier_init();')
        body=body.replace('mw_wide_contract_gp','mw_wide_contract_gp_tma')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}','-DFUSED_GP=1','-DPRE_TRANSPOSED=1']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_contract_gp_tma').kernel('mw_wide_contract_gp_tma')
        self.smem=57344+128;self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module();ab=p.front.ab
        def tm(x,channels):return launch.tensor_map(x,[64,64],dims=[n,channels*n],strides_bytes=[n*2],swizzle='128B',l2='128B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=launch.Struct([*[tm(x,d) for x in aa],*[tm(x,d) for x in bb],plan.pre,plan.f.mask,*p.gp,p.dl,p.dr,tm(plan.pre,8*d),tm(plan.f.mask,1),*[tm(x,h) for x in p.gp],n])
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//64)**2,1,1),(128,1,1),[self.params],self.smem)
