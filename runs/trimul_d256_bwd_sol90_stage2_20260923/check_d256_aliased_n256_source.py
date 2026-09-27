from pathlib import Path
import sys,os,json
from types import SimpleNamespace
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_aliased_n256_source import AliasedN256Source
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-aliased-n256-source-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();part=p.floats[7].clone()
    launch=T._launch_module()
    tm=lambda t:launch.tensor_map(t,[64,32],dims=[p.M,2*D],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
    params=launch.Struct([p.maps[0],W.tm(p.w1),tm(p.dl),tm(p.dr),*[tm(t) for t in p.gp],plan.f.mask,p.floats[7],p.M])
    proxy=SimpleNamespace(params=params,source_smem=3*128*D+16384+128,splits=plan.b7.splits)
    old=plan.b7.source_only
    for variant in (96,):
        try:op=AliasedN256Source(p,proxy,producer_regs=variant)
        except RuntimeError as exc:
            record['candidates'].append(dict(variant=variant,compile_failure=str(exc)))
            print('COMPILE_FAILURE',str(exc),flush=True)
            continue
        def baseline():plan.b7.source_only=old;return plan()
        def candidate():plan.b7.source_only=op;return plan()
        yn,gn=candidate();torch.cuda.synchronize()
        result=dict(variant=variant,cubin=str(op.cubin),gp_error=error(p.gp_all,gp),partial_error=error(p.floats[7],part))
        drv=op.k.unit.drv
        result['resident_ctas']=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(op.k.handle)),256,op.smem)))
        result['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        result['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in result['errors'].items())
        print('CHECK',result,flush=True)
        if result['strict']:
            result['times']=paired(dict(baseline_source=old,new_source=op,baseline_full=baseline,new_full=candidate))
            print('TIMES',variant,{k:v['median_us'] for k,v in result['times'].items()},flush=True)
        record['candidates'].append(result);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
