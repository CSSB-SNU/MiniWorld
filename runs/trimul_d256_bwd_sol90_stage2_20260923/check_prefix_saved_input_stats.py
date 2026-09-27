from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training
from prefix_saved_input_stats import SavedInputReduce
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-prefix-saved-input-stats-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];old=plan.dx.reduce_only
    op=SavedInputReduce(p,old,splits=plan.b7.splits)
    item=dict(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy);record['candidates'].append(item)
    def before():plan.dx.reduce_only=old;return plan()
    def after():plan.dx.reduce_only=op;return plan()
    yn,gn=after();torch.cuda.synchronize()
    item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
    print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
    if item['strict']:
        item['stage_times']=paired(dict(old_finish=old,new_finish=op));gc.collect()
        item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in item[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
