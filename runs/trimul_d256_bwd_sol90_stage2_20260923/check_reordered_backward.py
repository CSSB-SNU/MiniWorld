from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from reordered_backward import ReorderedBackward
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[i//2];N=(384,768)[i%2]
if D==256:from d256_stats_checkpoint import Training
else:from wide_checkpoint16 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-reordered-backward-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);y,g=plan();expected=[x.clone() for x in [y,*g]]
    def before():return plan.forward(),plan.backward()
    for mode in ('dw_first','ln_first','dw_after_source','dw_last'):
        op=ReorderedBackward(plan,mode)
        def after():return plan.forward(),op()
        yn,gn=after();torch.cuda.synchronize()
        item=dict(mode=mode,errors={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)})
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        record['candidates'].append(item);print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['backward_times']=paired(dict(old_backward=plan.backward,new_backward=op));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',mode,{k:v['median_us'] for key in ('backward_times','full_times') for k,v in item[key].items()},flush=True)
        before();path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
