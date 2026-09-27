import json
import sys
from pathlib import Path
import torch
from miniworld_engine.autotune import cache
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.rmsnorm.triton.main import rmsnorm_fwd_kernel,rmsnorm_bwd_kernel

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'triton_audit'))
from audit import measure

record=json.loads((root.parent/'triton_audit/bench-1-0.json').read_text())
m,n=record['M'],record['D']
x=torch.randn(m,n,device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
dy=torch.randn_like(x);dx=torch.empty_like(x);w=torch.randn(n,device='cuda')
dw=torch.zeros_like(w);inv=torch.empty(m,device='cuda')
bound=dict(X=x,Y=y,W=w,Rstd=inv,DX=dx,DY=dy,DW=dw,
           stride_r=n,stride_c=1,M=m,N=n,eps=1e-5,shape_key=both_key(m,N=n),HAS_WEIGHT=True)
for stage,tuner in [('fwd',rmsnorm_fwd_kernel),('bwd',rmsnorm_bwd_kernel)]:
    old=record['configs'][f'rmsnorm_{stage}_kernel']
    cfg=next(c for c in tuner.configs if c.kwargs==old['kwargs'] and c.num_warps==old['warps'] and c.num_stages==old['stages'])
    bm,bk=cfg.kwargs['BLOCK_M1'],cfg.kwargs['BLOCK_K']
    def call():
        if stage=='fwd':
            return tuner.fn[((m+bm-1)//bm,)](x,y,w,inv,n,1,m,n,1e-5,bm,bk,0,True,num_warps=cfg.num_warps,num_stages=cfg.num_stages)
        dw.zero_()
        return tuner.fn[((m+bm-1)//bm,)](dx,dy,dw,x,w,inv,n,1,m,n,bm,bk,0,True,num_warps=cfg.num_warps,num_stages=cfg.num_stages)
    timing=measure(call)['median_ms']
    op=f'rmsnorm_{stage}_triton'
    cache.store_ranked_configs(op,cache.gpu_key(),cache.dtype_of_args(bound),cache.bucket_of_autotuner(tuner,bound),
        [(cfg,timing)],cache.config_space_hash(tuner.configs),top_k=1,op_id=cache.op_identity(tuner),
        configs=tuner.configs,entry_configs=[cfg],measurement=cache.measurement_workload(op,tuner,bound))
    print(stage,timing,cfg,flush=True)
