"""Keep one WGMMA group outstanding while loading the next contraction tile."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class AsyncContractGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        body=prior.source_text
        start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
        body=body[:start]+''' int tiles=p.N/128;
 int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'''+body[end:]
        prefetch='  if(threadIdx.x==0 && ki+128<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS);'
        assert body.count(prefetch)==1
        body=body.replace(prefetch,'')
        marker='  fence_regs(v);wgmma_fence();'
        assert body.count(marker)==1
        body=body.replace(marker,'  if(it==0){fence_regs(v);wgmma_fence();}')
        marker='  });wgmma_commit();wgmma_wait<0>();fence_regs(v);'
        assert body.count(marker)==1
        body=body.replace(marker,'  });wgmma_commit();wgmma_wait<1>();\n  __syncthreads();\n'+prefetch)
        marker=' // Both consumers finish'
        assert body.count(marker)==1
        body=body.replace(marker,' wgmma_wait<0>();fence_regs(v);\n'+marker)
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_async_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_async_contract_gp').kernel('mw_wide_async_contract_gp');self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
