from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint12 import Training
from wide_stream_dn_ln import StreamDnLN as FusedDnLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-stream-dn-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dn=p.tensors[9].clone();dt=p.dt.clone()
    diagnostic=FusedDnLN(plan,True);diagnostic();torch.cuda.synchronize()
    record.update(dn_error=error(p.tensors[9],dn),dt_error=error(p.dt,dt),diagnostic_cubin=str(diagnostic.cubin))
    op=FusedDnLN(plan,False)
    import subprocess
    print(subprocess.run(['cuobjdump','--dump-resource-usage',str(op.cubin)],capture_output=True,text=True).stdout,flush=True)
    oldrun=plan.schedule.run;oldln=plan.ln
    def newrun(name):
        if name!='dn':oldrun(name)
    def baseline():plan.schedule.run=oldrun;plan.ln=oldln;return plan()
    def candidate():plan.schedule.run=newrun;plan.ln=op;return plan()
    yn,gn=candidate();torch.cuda.synchronize();record['cubin']=str(op.cubin);record['grid']=op.rows
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    def oldboth():oldrun('dn');oldln()
    if record['strict']:
        record['times']=paired(dict(baseline_dn_ln=oldboth,new_dn_ln=op,baseline_full=baseline,new_full=candidate))
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
