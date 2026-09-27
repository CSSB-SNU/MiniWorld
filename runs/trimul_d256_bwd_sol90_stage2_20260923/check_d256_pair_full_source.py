from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_lt_checkpoint import Training
from d256_pair_full_source import PairFullSource
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,variants=[])
path=THIS/f'result-d256-pair-full-source-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]]
    old=plan.b7.source_only;gp=p.gp_all.clone();partial=p.floats[7].clone()
    for full_n,bulk in ((False,True),(True,True),(True,False)):
        plan.b7.source_only=old;plan()
        op=PairFullSource(p,plan.b7,full_n,bulk);plan.b7.source_only=op
        yn,gn=plan();torch.cuda.synchronize()
        result=dict(full_n=full_n,bulk=bulk,cubin=str(op.cubin),gp_error=error(p.gp_all,gp),partial_error=error(p.floats[7],partial))
        result['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        result['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in result['errors'].items())
        print('CHECK',result,flush=True)
        def baseline():plan.b7.source_only=old;return plan()
        def candidate():plan.b7.source_only=op;return plan()
        if result['strict']:
            result['times']=paired(dict(baseline_source=old,new_source=op,baseline_full=baseline,new_full=candidate))
            print('TIMES',{k:v['median_us'] for k,v in result['times'].items()},flush=True)
        record['variants'].append(result);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
