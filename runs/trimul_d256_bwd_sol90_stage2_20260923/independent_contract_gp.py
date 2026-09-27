"""Let the two compute groups retire K tiles independently using slot credits."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_two_group_contract_gp import TwoGroupContractGP

class IndependentContractGP(TwoGroupContractGP):
    def __init__(self,plan):
        super().__init__(plan,channel_group=plan.channel_group)
        body=self.source_text
        point='int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();'
        assert body.count(point)==1
        body=body.replace(point,'int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);named_bar_sync(1+WG,128);')
        point='if(threadIdx.x==0 && ki+128<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS);'
        assert body.count(point)==1
        body=body.replace(point,'''if(threadIdx.x==0 && ki+128<p.N){
   if(it>=1)mbar_wait(bar+SLOTS+(it+2)%SLOTS,((it+2)/SLOTS-1)&1);
   load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS);
  }
  named_bar_sync(1+WG,128);''')
        point='  });wgmma_commit();wgmma_wait<0>();fence_regs(v);'
        assert body.count(point)==1
        body=body.replace(point,point+'\n  named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);')
        body=body.replace('mw_wide_two_group_contract_gp','mw_independent_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_independent_contract_gp').kernel('mw_independent_contract_gp');self.k.set_max_dynamic_smem(self.smem)
