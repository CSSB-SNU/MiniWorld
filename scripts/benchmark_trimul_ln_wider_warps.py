"""Check whether the canonical LN backward's 1/2-warp grid excludes faster schedules."""
import argparse
import itertools
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch,triton
    from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd as mod
    from miniworld_engine.autotune.shape_key import both_key,pack
    reports=[]
    for length in [384,768]:
        m,n=length**2,256
        torch.manual_seed(78)
        x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t()
        dy=torch.randn_like(x);g=torch.randn(n,device='cuda')
        mean=x.float().mean(1);rs=torch.rsqrt(x.float().var(1,unbiased=False)+1e-5)
        ref=mod._ln_bwd_atomic(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
        atomic=lambda:mod._ln_bwd_atomic(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
        canonical=lambda:mod._ln_bwd_persistent_canonical(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
        baseline=dict(atomic=triton.testing.do_bench_cudagraph(atomic,rep=40),
                      canonical=triton.testing.do_bench_cudagraph(canonical,rep=40))
        np=mod._persistent_grid(x.device)
        dx=torch.empty_like(x);dg=torch.empty(np,n,device='cuda');db=torch.empty_like(dg)
        rows=[]
        for bm,bk,warps in itertools.product([16,32,64,128],[128,256],[4,8]):
            cfg=dict(BLOCK_M1=bm,BLOCK_K=bk,num_warps=warps,num_stages=3)
            def call():
                mod._ln_bwd_persistent_jit.fn[(np,triton.cdiv(n,bk))](
                    dx,dg,db,dy,x,g,mean,rs,n,*x.stride(),m,N=n,
                    shape_key=pack(both_key(m),N=n),**cfg)
                return dx,dg.sum(0),db.sum(0)
            try:
                out=call()
                errors=[float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip(out,ref)]
                assert max(errors)<.002,errors
                ms=triton.testing.do_bench_cudagraph(call,rep=8)
                rows.append(dict(config=cfg,ms=ms,errors=errors))
            except triton.OutOfResources as error:
                rows.append(dict(config=cfg,error=str(error)))
        report=dict(length=length,baseline=baseline,rows=rows,
                    best=min((r for r in rows if 'ms' in r),key=lambda r:r['ms']))
        reports.append(report);print('RESULT',length,baseline,report['best'],flush=True)
    args.output.write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
