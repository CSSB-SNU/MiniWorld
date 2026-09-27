"""Test GP store eviction priority while preserving arithmetic and tiling."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class GPStorePolicy:
    def __init__(self,plan,policy):
        assert policy in ('first','last')
        p=plan.p;self.p=p;self.wide=p.D!=256
        prior=plan.contract_gp if self.wide else plan.b7
        self.__dict__.update(prior.__dict__)
        if self.wide:
            body=prior.source_text
            start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
            body=body[:start]+' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'+body[end:]
            oldname='mw_wide_two_group_contract_gp';self.threads=256;self.grid=4*p.D*(p.n//128)**2
            define=f'-DWIDTH={p.D}'
        else:
            body=mask_stage(prior.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
            oldname='mw_d256_b7_tma';self.threads=prior.source_threads;self.grid=32*prior.splits
            define=f'-DWEIGHT_SPLITS={prior.splits}'
        marker='cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];'
        assert body.count(marker)==1,body.count(marker)
        body=body.replace(marker,'{.reg .b64 policy;createpolicy.fractional.L2::evict_'+policy+'.b64 policy,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],policy;}')
        body=body.replace(oldname,'mw_gp_store_policy')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),define]
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_gp_store_policy').kernel('mw_gp_store_policy');self.kernel.set_max_dynamic_smem(self.smem)
    def __call__(self):self.kernel.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
