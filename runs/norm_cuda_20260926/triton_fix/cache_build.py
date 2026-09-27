"""Store only actually measured RMS configurations; preserve partial coverage."""
import json
from pathlib import Path
import torch
import triton
from miniworld_engine.autotune import cache
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.rmsnorm.triton.main import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel

root = Path(__file__).parent
data = json.loads((root.parent / 'triton_audit/tile-probe.json').read_text())
torch.cuda.init()
for row in data:
    m, n = row['M'], row['D']
    x = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    y, dx, dy = (torch.empty_like(x) for _ in range(3))
    w, dw = (torch.empty(n, device='cuda') for _ in range(2))
    inv = torch.empty(m, device='cuda')
    for stage, tuner in [('fwd', rmsnorm_fwd_kernel), ('bwd', rmsnorm_bwd_kernel)]:
        op = f'rmsnorm_{stage}_triton'
        ranked = []
        for trial in row['trials'][stage]:
            if 'error' in trial:
                continue
            c = trial['config']
            cfg = triton.Config({'BLOCK_M1':c['bm'], 'BLOCK_K':c['bk']}, num_warps=c['warps'], num_stages=c['stages'])
            matches = [q for q in tuner.configs if cache._sig(q) == cache._sig(cfg)]
            if matches:
                ranked.append((matches[0], trial['timing']['median_ms']))
        assert ranked
        ranked.sort(key=lambda v: v[1])
        bound = dict(X=x,Y=y,W=w,Rstd=inv,DX=dx,DY=dy,DW=dw,
                     stride_r=n,stride_c=1,M=m,N=n,eps=1e-5,
                     shape_key=both_key(m,N=n), HAS_WEIGHT=True)
        path = cache.store_ranked_configs(
            op,cache.gpu_key(),cache.dtype_of_args(bound),
            cache.bucket_of_autotuner(tuner,bound),ranked,cache.config_space_hash(tuner.configs),
            top_k=3,op_id=cache.op_identity(tuner),configs=tuner.configs,
            entry_configs=[c for c,t in ranked],measurement=cache.measurement_workload(op,tuner,bound))
        print(op,m,n,len(ranked),str(path),flush=True)
