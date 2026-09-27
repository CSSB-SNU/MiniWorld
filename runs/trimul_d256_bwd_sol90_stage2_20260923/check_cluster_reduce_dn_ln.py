from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from cluster_reduce_dn_ln import ClusterReduceDnLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','2'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-cluster-reduce-dn-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dn=p.tensors[9].clone();dt=p.dt.clone()
    diag=ClusterReduceDnLN(plan,True);print('DIAGNOSTIC_LAUNCH',str(diag.cubin),diag.registers,diag.local_bytes,diag.active_clusters,flush=True)
    p.tensors[9].fill_(float('nan'));p.dt.fill_(float('nan'));diag();torch.cuda.synchronize()
    record.update(dn_error=error(p.tensors[9],dn),dt_error=error(p.dt,dt),diagnostic_cubin=str(diag.cubin))
    print('DIAGNOSTIC',record,flush=True);path.write_text(json.dumps(record,indent=2))
    op=ClusterReduceDnLN(plan);oldrun=plan.schedule.run;oldln=plan.ln
    record.update(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,active_clusters=op.active_clusters)
    def newrun(name):
        if name!='dn':oldrun(name)
    def before():plan.schedule.run=oldrun;plan.ln=oldln;return plan()
    def after():plan.schedule.run=newrun;plan.ln=op;return plan()
    yn,gn=after();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    def oldstage():oldrun('dn');oldln()
    if record['strict']:
        record['stage_times']=paired(dict(old_dn_ln=oldstage,new_dn_ln=op));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
