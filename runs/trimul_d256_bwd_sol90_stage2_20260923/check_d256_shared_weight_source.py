from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_stats_checkpoint import Training
from d256_shared_weight_source import SharedWeightSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-shared-weight-source-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    part=p.floats[7].reshape(-1)[3*D*D:].as_strided((plan.b7.splits,8*D*D),(11*D*D,1));partial=part.clone()
    old=plan.b7.source_only;op=SharedWeightSource(plan.b7)
    def before():plan.b7.source_only=old;return plan()
    def after():plan.b7.source_only=op;return plan()
    yn,gn=after();torch.cuda.synchronize()
    record['gp_error']=error(p.gp_all,gp);record['partial_error']=error(part,partial)
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    record['registers']=op.registers;record['local_bytes']=op.local_bytes;record['cubin']=str(op.cubin);print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        record['stage_times']=paired(dict(old_source=old,new_source=op));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
