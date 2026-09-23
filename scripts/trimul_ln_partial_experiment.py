"""Experiment: output LN backward with per-tile partials instead of contended atomics."""
import argparse,itertools,json
from pathlib import Path
import torch,triton
import triton.language as tl

@triton.jit
def partial_kernel(DXn, X, G, Mean, Rstd, DX, DG, DB, M, N,
                   sdn0, sdn1, sx0, sx1, sdx0, sdx1,
                   BLOCK_M1: tl.constexpr, N_PAD: tl.constexpr, shape_key,
                   ):
    # NOTE: one tile per program (atomic_add per block for dγ/dβ). A grid-stride variant (one
    # atomic per program) sped up CONTIGUOUS large-d (d512 0.96→1.07x) but CATASTROPHICALLY
    # regressed m-major d=256 (2.45→0.45x — the strided x/dx access interacts badly with the
    # strided loop), so it was reverted. Keep this simple form (good on both layouts).
    row = tl.program_id(0).to(tl.int64)
    rm = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    rmask = rm < M
    cols = tl.arange(0, N_PAD)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    dxn = tl.load(DXn + rm[:, None] * sdn0 + cols[None, :] * sdn1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + rm, mask=rmask, other=0.0)[:, None]
    rstd = tl.load(Rstd + rm, mask=rmask, other=0.0)[:, None]
    g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    xhat = tl.where(cmask[None, :], (x - mean) * rstd, 0.0)
    dxhat = dxn * g
    inv_n = 1.0 / N
    c2 = tl.sum(tl.where(cmask[None, :], dxhat, 0.0), axis=1) * inv_n        # meanₖ(dx̂)
    c1 = tl.sum(tl.where(cmask[None, :], dxhat * xhat, 0.0), axis=1) * inv_n  # meanₖ(dx̂·x̂)
    dx = rstd * (dxhat - c2[:, None] - xhat * c1[:, None])
    tl.store(DX + rm[:, None] * sdx0 + cols[None, :] * sdx1,
             dx.to(DX.dtype.element_ty), mask=mask)
    pdg = tl.sum(tl.where(mask, dxn * xhat, 0.0), axis=0)
    pdb = tl.sum(tl.where(mask, dxn, 0.0), axis=0)
    tl.store(DG + row * N + cols, pdg, mask=cmask)
    tl.store(DB + row * N + cols, pdb, mask=cmask)


def install_from_report(path):
    """Experimental module A/B only; the production package is not modified."""
    from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd as mod
    config=json.loads(Path(path).read_text())['best']['config']
    previous=mod._ln_bwd_atomic

    def candidate(dxn,x,gamma,mean,rstd,dx_strides,*,shape_key=None):
        if tuple(x.shape)!=(384**2,256):
            return previous(dxn,x,gamma,mean,rstd,dx_strides,shape_key=shape_key)
        m,n=x.shape
        dx=torch.empty_strided((m,n),dx_strides,device=x.device,dtype=dxn.dtype)
        blocks=triton.cdiv(m,config['BLOCK_M1'])
        dg=torch.empty((blocks,n),device=x.device,dtype=torch.float32)
        db=torch.empty_like(dg)
        partial_kernel[(blocks,)](dxn,x,gamma,mean,rstd,dx,dg,db,m,n,
                                  *dxn.stride(),*x.stride(),*dx.stride(),
                                  N_PAD=triton.next_power_of_2(n),shape_key=shape_key or 0,**config)
        return dx,dg.sum(0),db.sum(0)

    mod._ln_bwd_atomic=candidate
    return previous


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from miniworld_engine.kernels.layernorm_linear.triton.mmajor_bwd import _ln_bwd_atomic
    from miniworld_engine.autotune.shape_key import both_key
    m,n=384**2,256
    torch.manual_seed(69)
    x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t()
    dy=torch.randn_like(x);g=torch.randn(n,device='cuda')
    mean=x.float().mean(1);rs=torch.rsqrt(x.float().var(1,unbiased=False)+1e-5)
    original=lambda:_ln_bwd_atomic(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
    ref=original()
    baseline=triton.testing.do_bench_cudagraph(original,rep=80)
    rows=[]
    for bm,warps,stages in itertools.product([16,32,64,128],[2,4,8,16],[1,2,3]):
        blocks=triton.cdiv(m,bm)
        dx=torch.empty_like(x)
        dg=torch.empty(blocks,n,device='cuda');db=torch.empty_like(dg)
        def call():
            partial_kernel[(blocks,)](dy,x,g,mean,rs,dx,dg,db,m,n,
                                      *dy.stride(),*x.stride(),*dx.stride(),
                                      BLOCK_M1=bm,N_PAD=triton.next_power_of_2(n),shape_key=both_key(m),
                                      num_warps=warps,num_stages=stages)
            return dx,dg.sum(0),db.sum(0)
        cfg=dict(BLOCK_M1=bm,num_warps=warps,num_stages=stages)
        try:
            out=call()
            errors=[float((v.float()-r.float()).norm()/r.float().norm()) for v,r in zip(out,ref)]
            assert max(errors)<.002,errors
            ms=triton.testing.do_bench_cudagraph(call,rep=12)
            rows.append(dict(config=cfg,ms=ms,errors=errors))
        except triton.OutOfResources as exc:
            rows.append(dict(config=cfg,error=str(exc)))
    report=dict(baseline_ms=baseline,rows=rows,best=min((r for r in rows if 'ms' in r),key=lambda r:r['ms']))
    args.output.write_text(json.dumps(report,indent=2));print('RESULT',baseline,report['best'],flush=True)


if __name__=='__main__':main()
