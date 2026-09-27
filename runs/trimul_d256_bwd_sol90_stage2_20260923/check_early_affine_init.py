"""Full-workload pilot: output-affine zeroing moves before LN in the stream."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from early_affine_init import EarlyAffineGate,IndependentOutputLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//3];scale=(1,2,4)[index%3];N=384
if D==256:from d256_contract_checkpoint import Training
else:from wide_checkpoint21 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,grid_scale=scale,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-early-affine-init-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone()
    owner=plan.b1 if D==256 else plan;oldln=owner.ln;oldgate=plan.prefix_gate
    gate=EarlyAffineGate(plan);ln=IndependentOutputLN(p,oldln,scale)
    def select(enabled):
        owner.ln=ln if enabled else oldln
        plan.prefix_gate=gate if enabled else oldgate
        if D==256:plan.b1.prepare=plan.prefix_gate
        else:plan.b1.epi=plan.prefix_gate
    def before():select(False);return plan()
    def after():select(True);return plan()
    for i in (10,11):p.floats[i].fill_(float('nan'))
    yn,gn=after();torch.cuda.synchronize()
    record.update(dt_error=error(p.dt,dt),registers=ln.registers,local_bytes=ln.local_bytes,grid=ln.grid,cubins=[str(gate.cubin),str(ln.cubin)])
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not ln.local_bytes:
        def oldstage():oldgate();oldln()
        def newstage():gate();ln()
        record['stage_times']=paired(dict(old_gate_ln=oldstage,new_gate_ln=newstage));gc.collect()
        def oldbw():select(False);return plan.backward()
        def newbw():select(True);return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['stage_times','backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
