"""Two K slots and register mask loads permit three resident fused CTAs."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_two_group_contract_gp import TwoGroupContractGP
class CompactContractGP(TwoGroupContractGP):
    def __init__(self,plan,minblocks=3):
        super().__init__(plan,channel_group=plan.channel_group)
        body=self.source_text
        body=body.replace('SLOTS=3','SLOTS=2')
        body=body.replace('load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);','')
        body=body.replace('ki+128<p.N','ki+64<p.N').replace('ki+128,(it+2)%SLOTS','ki+64,(it+1)%SLOTS')
        body=body.replace('mbar_arrive_expect_tx(bar+6,98304)','mbar_arrive_expect_tx(bar+6,65536)')
        point='   tma_load_2d(sm+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);'
        assert body.count(point)==1;body=body.replace(point,'')
        body=body.replace('mask=*reinterpret_cast<uint32_t*>(sm+65536+off)','mask=*reinterpret_cast<const uint32_t*>(p.mask+size_t(mi+WG*64+r)*p.N+ni+c)')
        body=body.replace('__launch_bounds__(256,2)',f'__launch_bounds__(256,{minblocks})')
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_compact_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_compact_contract_gp').kernel('mw_wide_compact_contract_gp')
        self.smem=65536+128;self.k.set_max_dynamic_smem(self.smem)
