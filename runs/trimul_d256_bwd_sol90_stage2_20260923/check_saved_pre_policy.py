from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint15 import Training
from wide_saved_pre_policy import SavedPrePolicy
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-saved-pre-policy-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    oldfront=plan.f.front;pre=plan.pre.clone()
    for policy in ('first','last'):
        op=SavedPrePolicy(plan.baseline_front,plan.pre,policy);item=dict(policy=policy,cubin=str(op.cubin));record['candidates'].append(item)
        def before():plan.f.front=oldfront;return plan()
        def after():plan.f.front=op;return plan()
        plan.pre.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize();item['pre_error']=error(plan.pre,pre)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['stage_times']=paired(dict(old_front=oldfront,new_front=op));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
