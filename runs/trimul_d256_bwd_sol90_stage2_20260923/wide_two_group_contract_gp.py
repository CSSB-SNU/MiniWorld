"""Two compute warpgroups keep two CTAs resident for the fused contraction."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_contract_gp_tma import TmaContractGP

class TwoGroupContractGP(TmaContractGP):
    def __init__(self,plan,fused=True,transposed=True,channel_group=32):
        super().__init__(plan,fused,transposed)
        root=Path(__file__).resolve().parent
        body=(root/'wide_pipe_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        helper='''
template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g,n=ni+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
  tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
 }
}
'''
        begin=body.index('template<int MODE> TMN_DEVI void produce(')
        end=body.index('template<int MODE,int WG> TMN_DEVI void consume(',begin)
        producer=body[begin:end]
        epilogue_load=producer[producer.index(' // Both consumers finish'):]
        epilogue_load=epilogue_load[:epilogue_load.rindex('}')]
        body=body[:begin]+helper+body[end:]
        point=' int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float v[64]={};'
        assert body.count(point)==1
        body=body.replace(point,point+'''\n if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);}
''')
        point='int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);named_bar_sync(1+WG,128);'
        assert body.count(point)==1
        body=body.replace(point,'''int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && ki+128<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS);''')
        point='  named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);'
        assert body.count(point)==1;body=body.replace(point,'')
        point=' __syncthreads();mbar_wait(bar+6,0);__syncthreads();'
        assert body.count(point)==1;body=body.replace(point,epilogue_load)
        point=''' if(threadIdx.x<128){setmaxnreg_dec<32>();produce<MODE>(p,sm,bar,ch,mi,ni);}
 else{setmaxnreg_inc<232>();if(threadIdx.x<256)consume<MODE,0>(p,sm,bar,ch,mi,ni);else consume<MODE,1>(p,sm,bar,ch,mi,ni);}'''
        assert body.count(point)==1
        body=body.replace(point,' consume<MODE>(p,sm,bar,ch,mi,ni);')
        body=body.replace('template<int MODE,int WG> TMN_DEVI void consume','template<int MODE> TMN_DEVI void consume')
        point=' int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float v[64]={};'
        assert body.count(point)==1
        body=body.replace(point,' const int WG=threadIdx.x/128;\n'+point)
        body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(256,2)').replace('mw_wide_pipe_contract_gp','mw_wide_two_group_contract_gp')
        point='int tile=(rem/32)%(tiles*tiles),ch=(rem%32)+32*(rem/(32*tiles*tiles));'
        assert body.count(point)==1 and channel_group in (1,2,4,8,16,32)
        body=body.replace(point,f'int tile=(rem/{channel_group})%(tiles*tiles),ch=(rem%{channel_group})+{channel_group}*(rem/({channel_group}*tiles*tiles));')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_two_group_contract_gp').kernel('mw_wide_two_group_contract_gp')
        self.smem=98304+128;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
