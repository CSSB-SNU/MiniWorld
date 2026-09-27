from pathlib import Path
import sys,os,json
from types import SimpleNamespace
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint7 import Training
from wide_two_group_contract_gp import TwoGroupContractGP
from wide_saved_front_transposed import SavedTransposedFront
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-contract-gp-order-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    pre=p.x.new_empty((8*D,p.M));oldfront=plan.f.front
    front=SavedTransposedFront(plan.baseline_front,pre);front();torch.cuda.synchronize()
    oldrun=plan.schedule.run;oldsource=plan.b7.source_only
    def fusedrun(name):
        if name not in ('bc0','bc1','bc2','bc3'):oldrun(name)
    def baseline():
        plan.f.front=oldfront;plan.schedule.run=oldrun;plan.b7.source_only=oldsource;return plan()
    def oldboth():
        for name in ('bc0','bc1','bc2','bc3'):oldrun(name)
        oldsource.derivatives()
    for group in (1,2,4,8,16):
        fused=TwoGroupContractGP(SimpleNamespace(p=p,pre=pre,f=plan.f),channel_group=group)
        def fusedsource():fused();plan.saved_source.matmul()
        def candidate():
            plan.f.front=front;plan.schedule.run=fusedrun;plan.b7.source_only=fusedsource;return plan()
        yn,gn=candidate();torch.cuda.synchronize()
        result=dict(group=group,cubin=str(fused.cubin),gp_error=error(p.gp_all,gp))
        result['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        result['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in result['errors'].items())
        print('CHECK',result,flush=True)
        if result['strict']:
            result['times']=paired(dict(baseline_contract_gp=oldboth,new_contract_gp=fused,baseline_full=baseline,new_full=candidate))
            print('TIMES',group,{k:v['median_us'] for k,v in result['times'].items()},flush=True)
        record['candidates'].append(result);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
