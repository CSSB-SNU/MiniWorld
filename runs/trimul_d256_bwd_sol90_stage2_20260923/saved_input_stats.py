"""Reuse our forward's existing input-LN statistics; experimental only."""
import os,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SavedInputFront:
    def __init__(self, original, stats):
        self.__dict__.update(original.__dict__)
        assert self.normalize
        a,b,slots,sk,mb=self.cfg[:5]
        shared=self.cfg[5] if len(self.cfg)>5 else 0
        stream=int(shared==2)
        include='front_shared.cuh' if shared else 'tmn_kernels.cuh'
        func='mw_k1_shared' if shared else 'k1_body'
        body=(f'#include "{include}"\n'
              f'using C=tmn::K1Cfg<256,512,false,{a},{b},{slots},{sk},{1 if stream else -1}>;\n'
              'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
              'void mw_d256_front_save_stats(__grid_constant__ const tmn::K1Params p){'
              f'tmn::sm90::{func}<C,true,1,true,true,1>(p);}}')
        flags=[f'-DMW_MINB={mb}',f'-DMW_K1_STREAM={stream}',
               '-DTMN_SIGMOID_TANH=1','-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1',
               '-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-I'+str(F.R)]
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_front_save_stats').kernel('mw_d256_front_save_stats')
        self.k.set_max_dynamic_smem(self.smem)
        fields=original.params.fields.copy();assert fields[7] is None;fields[7]=stats
        self.params=T._launch_module().Struct(fields)

    def __call__(self):
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)


def enable(plan, gamma_cache=False):
    from dx_ln import DxLN
    assert all(os.environ.get(k,'0')=='0' for k in ('DX_N256','DX_PIPE','DX_SPLIT','DX_IN_STATS','DX_GAMMA_CACHE'))
    if not isinstance(plan.f.front,SavedInputFront):
        plan.p.input_stats=torch.empty((plan.p.M,2),device=plan.p.x.device,dtype=torch.float32)
        plan.f.front=SavedInputFront(plan.f.front,plan.p.input_stats)
        plan.saved=(*plan.saved,plan.p.input_stats)
    prior=os.environ.get('DX_IN_STATS')
    prior_gamma=os.environ.get('DX_GAMMA_CACHE')
    try:
        os.environ['DX_IN_STATS']='1'
        os.environ['DX_GAMMA_CACHE']='1' if gamma_cache else '0'
        plan.b7.wide_finish=DxLN(plan.p,plan.b7.splits)
    finally:
        if prior is None:os.environ.pop('DX_IN_STATS',None)
        else:os.environ['DX_IN_STATS']=prior
        if prior_gamma is None:os.environ.pop('DX_GAMMA_CACHE',None)
        else:os.environ['DX_GAMMA_CACHE']=prior_gamma
    return dict(name='input_stats_gamma' if gamma_cache else 'input_stats',front_cubin=str(plan.f.front.cubin),
                cubin=str(plan.b7.wide_finish.cubin),stats_bytes=plan.p.input_stats.numel()*4,
                smem=plan.b7.wide_finish.smem,grid=plan.b7.wide_finish.grid)
