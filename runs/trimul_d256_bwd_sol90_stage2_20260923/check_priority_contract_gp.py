from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint8 import Training
from priority_contract_gp import PriorityContractGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-priority-contract-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    old=plan.contract_gp
    for group in ('stream','retain_inputs','both'):
        op=PriorityContractGP(plan,mode=group)
        def baseline():plan.contract_gp=old;return plan()
        def candidate():plan.contract_gp=op;return plan()
        yn,gn=candidate();torch.cuda.synchronize()
        result=dict(group=group,cubin=str(op.cubin),gp_error=error(p.gp_all,gp))
        result['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        result['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in result['errors'].items())
        print('CHECK',result,flush=True)
        if result['strict']:
            result['times']=paired(dict(baseline_contract_gp=old,new_contract_gp=op,baseline_full=baseline,new_full=candidate))
            print('TIMES',group,{k:v['median_us'] for k,v in result['times'].items()},flush=True)
        record['candidates'].append(result);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
