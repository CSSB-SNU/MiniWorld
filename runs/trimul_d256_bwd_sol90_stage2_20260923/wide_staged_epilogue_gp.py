"""Fill retired MMA slots with epilogue inputs during the final K steps."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class StagedEpilogueGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        assert self.p.n in (384,768) and self.p.n%192==0
        body=prior.source_text
        start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
        body=body[:start]+' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'+body[end:]
        helper='''
template<int MODE,int PLANE> TMN_DEVI void load_epi(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if constexpr(PLANE==0)mbar_arrive_expect_tx(bar+6,98304);
 for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
  int q=(wm*2+wn)*8192;
  if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
  else tma_load_2d(sm+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);
 }
}
'''
        marker='template<int MODE> TMN_DEVI void consume('
        assert body.count(marker)==1
        body=body.replace(marker,helper+marker)
        marker='int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
  // N is divisible by 3*64: slot0/slot1 retire before the last two steps.
  if(threadIdx.x==0){
   if(ki==p.N-128)load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
   if(ki==p.N-64)load_epi<MODE,1>(p,sm,bar,ch,mi,ni);
  }
''')
        begin=body.index(' // Both consumers finish all WGMMA reads')
        end=body.index(' mbar_wait(bar+6,0);__syncthreads();',begin)
        body=body[:begin]+''' // Both compute groups have now retired slot2.
 __syncthreads();
 if(threadIdx.x==0)load_epi<MODE,2>(p,sm,bar,ch,mi,ni);
'''+body[end:]
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_staged_epilogue_gp')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_staged_epilogue_gp').kernel('mw_wide_staged_epilogue_gp')
        self.k.set_max_dynamic_smem(self.smem)

    def __call__(self):
        self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
