from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_bounded_affine_ln import BoundedAffineLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
if D==256:from d256_gate_checkpoint import Training
else:from wide_checkpoint14 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,variants=[])
path=THIS/f'result-bounded-affine-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone()
    oldln=plan.b1.ln if D==256 else plan.ln
    configs=[(64,64,4),(64,64,8)] if D==256 else [(64,96,4),(64,96,8),(80,80,4)]
    for main,aff,unroll in configs:
        item=dict(main=main,affine=aff,unroll=unroll);record['variants'].append(item)
        try:op=BoundedAffineLN(p,main,aff,unroll)
        except RuntimeError as exc:
            item['compile_error']=str(exc);print('COMPILE_FAIL',item,flush=True);continue
        item.update(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy)
        print('LAUNCH',item,flush=True)
        def old():
            if D==256:plan.b1.ln=oldln
            else:plan.ln=oldln
            return plan()
        def new():
            if D==256:plan.b1.ln=op
            else:plan.ln=op
            return plan()
        yn,gn=new();torch.cuda.synchronize()
        item['dt_error']=error(p.dt,dt);item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['times']=paired(dict(old_ln=oldln,new_ln=op,old_full=old,new_full=new))
            print('TIMES',{k:v['median_us'] for k,v in item['times'].items()},flush=True)
        old()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
