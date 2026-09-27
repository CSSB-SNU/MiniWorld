"""Exact BF16-input sigmoid lookup for the contraction/GP epilogue."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class TableContractGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        self.table=torch.empty(65536,device=plan.p.x.device,dtype=torch.float32)
        body=prior.source_text
        body=body.replace('int N;};','int N;const float* sigmoid;};')
        marker='float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);'
        assert body.count(marker)==1
        body=body.replace(marker,'float ga=__ldg(p.sigmoid+(gr&65535u)),gb=__ldg(p.sigmoid+(gr>>16)),pa=bf16lo(pr),pb=bf16hi(pr);')
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_table_contract_gp')
        body+='''
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_build_sigmoid_table(float* out){
 unsigned i=blockIdx.x*256+threadIdx.x;out[i]=math::sigmoid(__uint_as_float(i<<16));
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        unit=T.load_unit(str(self.cubin),'mw_wide_table_contract_gp')
        self.k=unit.kernel('mw_wide_table_contract_gp');self.k.set_max_dynamic_smem(self.smem)
        self.init=unit.kernel('mw_wide_build_sigmoid_table');self.init.launch((256,1,1),(256,1,1),[self.table],0)
        self.params=T._launch_module().Struct(prior.params.fields+[self.table])
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
