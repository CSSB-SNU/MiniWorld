"""Keep reusable contraction inputs ahead of one-pass preactivation and GP traffic."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_two_group_contract_gp import TwoGroupContractGP

class PriorityContractGP(TwoGroupContractGP):
    def __init__(self,plan,mode='stream'):
        super().__init__(plan,channel_group=plan.channel_group)
        assert mode in ('stream','retain_inputs','both')
        body=self.source_text
        helper='''
template<bool RETAIN> TMN_DEVI void tma_priority(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){
 uint64_t policy;
 if constexpr(RETAIN)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(policy));
 else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1),"l"(policy):"memory");
}
'''
        point='template<int MODE> TMN_DEVI void load_input'
        assert body.count(point)==1;body=body.replace(point,helper+point)
        if mode in ('retain_inputs','both'):
            body=body.replace('tma_load_2d(sm+slot*INPUT','tma_priority<true>(sm+slot*INPUT')
        if mode in ('stream','both'):
            body=body.replace('tma_load_2d(sm+q,&p.premap','tma_priority<false>(sm+q,&p.premap')
            body=body.replace('tma_load_2d(sm+32768+q,&p.premap','tma_priority<false>(sm+32768+q,&p.premap')
            point='cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];'
            assert body.count(point)==1
            body=body.replace(point,'{.reg .b64 pol;createpolicy.fractional.L2::evict_first.b64 pol,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],pol;}')
        body=body.replace('mw_wide_two_group_contract_gp','mw_priority_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_priority_contract_gp').kernel('mw_priority_contract_gp')
        self.k.set_max_dynamic_smem(self.smem)
