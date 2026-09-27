"""Pilot: use existing AB plus exact correction codes instead of dense saved P."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ABProjectionGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        p=plan.p;n=p.n;d=p.D;self.plan=plan;root=Path(__file__).resolve().parent
        self.codes=torch.empty((4*d,n,n//4),device=p.x.device,dtype=torch.uint8)
        self.escapes=torch.empty((4*d,p.M),device=p.x.device,dtype=p.x.dtype)
        self.stats=torch.empty(5,device=p.x.device,dtype=torch.int64)
        body=prior.source_text
        marker='outmap[4];int N;';assert body.count(marker)==1
        body=body.replace(marker,'outmap[4],abmap,codemap;const bf* escapes;int N;')
        marker=' if constexpr(PLANE==0)mbar_arrive_expect_tx(bar+6,98304);';assert body.count(marker)==1
        body=body.replace(marker,''' if constexpr(PLANE==0){
  mbar_arrive_expect_tx(bar+6,98304+4096);
  tma_load_2d(sm+99328,&p.codemap,bar+6,ni/4,(SIDE*H+outch)*p.N+mi);
 }''')
        marker='if constexpr(PLANE<2)tma_load_2d';assert body.count(marker)==1
        body=body.replace(marker,'if constexpr(PLANE==0)tma_load_2d')
        marker='  else tma_load_2d(sm+65536+q,';assert body.count(marker)==1
        body=body.replace(marker,'''  else if constexpr(PLANE==1)tma_load_2d(sm+32768+q,&p.abmap,bar+6,ni+64*wn,(SIDE*H+outch)*p.N+mi+64*wm);
'''+marker)
        marker='template<int MODE> TMN_DEVI void load_input';assert body.count(marker)==1
        body=body.replace(marker,(root/'ab_projection_codec.cuh').read_text()+'\n'+marker)
        marker='  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);'
        assert body.count(marker)==1
        body=body.replace(marker,'''  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr));
  unsigned co=(WG*64+r)*32+c/4;co^=((co>>7)&1u)<<4;
  unsigned code=(sm[99328+co]>>(2*(c%4)))&15u;
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
  size_t ai=(size_t(SIDE*H+ch+HALF*D)*p.N+mi+64*WG+r)*p.N+ni+c;
  unsigned pa_raw=ab_projection_decode(pr&65535u,ga,mask&65535u,code&3u,p.escapes+ai);
  unsigned pb_raw=ab_projection_decode(pr>>16,gb,mask>>16,code>>2,p.escapes+ai+1);
  float pa=__uint_as_float(pa_raw<<16),pb=__uint_as_float(pb_raw<<16);
''')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_ab_projection_gp').replace('mw_wide_fixed_length_gp','mw_wide_ab_projection_gp')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-I'+str(root),f'-DWIDTH={d}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_ab_projection_gp').kernel('mw_wide_ab_projection_gp')
        self.smem=99328+4096;self.k.set_max_dynamic_smem(self.smem)
        self.pack_cubin=T.compile(root/'pack_ab_projection.cu',flags)
        self.pack=T.load_unit(str(self.pack_cubin),'mw_pack_ab_projection').kernel('mw_pack_ab_projection')
        L=T._launch_module()
        abmap=L.tensor_map(p.front.ab,[64,64],dims=[n,4*d*n],strides_bytes=[n*2],swizzle='128B',l2='128B')
        codemap=L.tensor_map(self.codes,[32,128],dims=[n//4,4*d*n],strides_bytes=[n//4],swizzle='32B',l2='128B')
        fields=prior.params.fields.copy();fields[-1:-1]=[abmap,codemap,self.escapes];self.params=L.Struct(fields)
        self.pack_args=[plan.pre,p.front.ab,plan.f.mask,self.codes,self.escapes,self.stats,L.i32(p.M),L.i32(d)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def encode(self):
        self.stats.zero_();self.pack.launch((1056,1,1),(256,1,1),self.pack_args,0)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
