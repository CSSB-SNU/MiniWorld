from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_async_full_width_dn_ln import AsyncFullWidthDnLN as FusedDnLN

os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
if D==256:from d256_permuted_ln_checkpoint import Training
else:from wide_checkpoint19 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-async-full-width-row64-dn-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dn=p.tensors[9].clone();dt=p.dt.clone()
    diagnostic=FusedDnLN(plan,True,64);diagnostic();torch.cuda.synchronize()
    record.update(registers=diagnostic.registers,local_bytes=diagnostic.local_bytes,occupancy=diagnostic.occupancy,dn_error=error(p.tensors[9],dn),dt_error=error(p.dt,dt),diagnostic_cubin=str(diagnostic.cubin))
    op=FusedDnLN(plan,False,64)
    import subprocess
    print(subprocess.run(['cuobjdump','--dump-resource-usage',str(op.cubin)],capture_output=True,text=True).stdout,flush=True)
    owner=plan.b1 if D==256 else plan
    oldrun=plan.schedule.run;oldln=owner.ln
    def newrun(name):
        if name!='dn':oldrun(name)
    def baseline():plan.schedule.run=oldrun;owner.ln=oldln;return plan()
    def candidate():plan.schedule.run=newrun;owner.ln=op;return plan()
    yn,gn=candidate();torch.cuda.synchronize();record['cubin']=str(op.cubin);record['grid']=op.rows;record['timed_registers']=op.registers;record['timed_local_bytes']=op.local_bytes
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    def oldboth():oldrun('dn');oldln()
    if record['strict'] and not op.local_bytes:
        record['times']=paired(dict(baseline_dn_ln=oldboth,new_dn_ln=op,baseline_full=baseline,new_full=candidate))
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
