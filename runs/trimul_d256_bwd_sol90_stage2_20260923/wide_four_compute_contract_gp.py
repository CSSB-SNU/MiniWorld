"""128x256 contraction: four compute groups share A/B tiles and GP TMA staging."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_two_group_contract_gp import TwoGroupContractGP

class FourComputeContractGP(TwoGroupContractGP):
    def __init__(self,plan,channel_group=1):
        super().__init__(plan,channel_group=channel_group)
        body=self.source_text
        body=body.replace('INPUT=32768,SLOTS=3,BAR=INPUT*SLOTS','INPUT=49152,SLOTS=3,BAR=196608')
        old=''' for(int g=0;g<2;++g){int m=mi+64*g,n=ni+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
  tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
 }'''
        new=''' for(int g=0;g<2;++g){int m=mi+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
 }
 for(int g=0;g<4;++g){int n=ni+64*g;
  tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
 }'''
        assert body.count(old)==1;body=body.replace(old,new)
        body=body.replace('sm+slot*INPUT+WG*8192','sm+slot*INPUT+(WG/2)*8192')
        body=body.replace('sm+slot*INPUT+16384),','sm+slot*INPUT+16384+(WG%2)*16384),')
        body=body.replace('mbar_arrive_expect_tx(bar+6,98304)','mbar_arrive_expect_tx(bar+6,196608)')
        body=body.replace('wn<2','wn<4').replace('(wm*2+wn)*8192','(wm*4+wn)*8192')
        body=body.replace('sm+65536+q','sm+131072+q').replace('sm+32768+q','sm+65536+q')
        body=body.replace('(WG*2+c/64)*8192','((WG/2)*4+(WG%2)*2+c/64)*8192')
        body=body.replace('sm+65536+off','sm+131072+off').replace('sm+32768+off','sm+65536+off')
        body=body.replace('sm+g*32768+','sm+g*65536+')
        start=body.index(' int tiles=p.N/128,mode=')
        end=body.index(' if(threadIdx.x==0)',start)
        body=body[:start]+f''' int nt=(p.N+255)/256,total=(p.N/128)*nt,mode=blockIdx.x/(D*total),rem=blockIdx.x%(D*total);
 int tile=(rem/{channel_group})%total,ch=(rem%{channel_group})+{channel_group}*(rem/({channel_group}*total));
 int mi=(tile/nt)*128,ni=(tile%nt)*256;
'''+body[end:]
        body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(512,1)').replace('mw_wide_two_group_contract_gp','mw_wide_four_compute_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_four_compute_contract_gp').kernel('mw_wide_four_compute_contract_gp')
        self.smem=196608+128;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)*((self.p.n+255)//256),1,1),(512,1,1),[self.params],self.smem)
