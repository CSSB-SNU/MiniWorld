from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_three_group_projection_dw import ThreeGroupProjectionDW
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D,splits=((384,11),(384,22),(384,44),(512,11),(512,22),(512,33))[index];N=384
if D==256:from d256_contract_checkpoint import Training
else:from wide_checkpoint20 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,splits=splits,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-three-group-projection-dw-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dwp=p.dwp.clone()
    op=ThreeGroupProjectionDW(plan,splits);oldrun=plan.schedule.run
    oldop=plan.late_output_dw.dwp if D==256 else lambda:oldrun('dwp')
    p.dwp.fill_(float('nan'));op.partial.fill_(float('nan'));op();torch.cuda.synchronize()
    record.update(dwp_error=error(p.dwp,dwp),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,cubin=str(op.cubin))
    def run(name):
        if name=='dwp':op()
        else:oldrun(name)
    def select(enabled):
        if D==256:plan.late_output_dw.dwp=op if enabled else oldop
        else:plan.schedule.run=run if enabled else oldrun
    def before():select(False);return plan()
    def after():select(True);return plan()
    yn,gn=after();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        record['stage_times']=paired(dict(old_dw=oldop,new_dw=op));gc.collect()
        def oldbw():select(False);return plan.backward()
        def newbw():select(True);return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['stage_times','backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
