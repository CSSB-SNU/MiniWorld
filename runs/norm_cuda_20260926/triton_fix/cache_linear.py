from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import cache
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.layernorm_linear.triton.fused import (
    layernorm_linear_triton_fwd, layernorm_linear_triton_fwd_stats, _lnl_fwd_kernel,
)

settings.configure(engine_backend='triton',autotune_miss_cap=24)
torch.set_num_threads(4)
for m,k,n in [(147456,128,16),(8192,384,512),(8192,64,64)]:
    x=torch.randn(1,m,k,device='cuda',dtype=torch.bfloat16)
    g=torch.randn(k,device='cuda');b=torch.randn_like(g)
    w=torch.randn(n,k,device='cuda',dtype=x.dtype)
    bias=torch.randn(n,device='cuda',dtype=x.dtype)
    for save,fn in [(False,layernorm_linear_triton_fwd),(True,layernorm_linear_triton_fwd_stats)]:
        result=fn(x,g,b,w,bias)
        if save:y,mean,inv=result
        else:
            y=result;mean=torch.empty(0,device='cuda');inv=torch.empty_like(mean)
        tuner=_lnl_fwd_kernel
        ranked=sorted([(c,t[0] if isinstance(t,(list,tuple)) else t) for c,t in tuner.configs_timings.items()],key=lambda ct:ct[1])
        ranked=[(c,t) for c,t in ranked if 0<t<float('inf')]
        assert ranked
        bound=dict(x_ptr=x.reshape(m,k),w_ptr=w,b_ptr=bias,g_ptr=g,beta_ptr=b,y_ptr=y.reshape(m,n),
                   M=m,N=n,K=k,eps=1e-5,stride_xm=k,stride_xk=1,stride_wn=k,stride_wk=1,
                   stride_ym=n,stride_yn=1,HAS_BIAS=True,shape_key=both_key(m,N=n,K=k),
                   Mean=mean,Rstd=inv,SAVE_STATS=save)
        op='layernorm_linear_fwd_triton'
        path=cache.store_ranked_configs(op,cache.gpu_key(),cache.dtype_of_args(bound),
            cache.bucket_of_autotuner(tuner,bound),ranked,cache.config_space_hash(tuner.configs),
            top_k=3,op_id=cache.op_identity(tuner),configs=tuner.configs,entry_configs=list(tuner.configs_timings),
            measurement=cache.measurement_workload(op,tuner,bound))
        print(m,k,n,save,len(ranked),path,flush=True)
