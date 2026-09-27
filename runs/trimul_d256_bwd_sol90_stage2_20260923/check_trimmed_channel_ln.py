from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training as D256Training
from wide_checkpoint14 import Training as WideTraining
from wide_trimmed_channel_ln import TrimmedChannelLN
os.environ.update(PREFIX_IMPL="blas",PREFIX_COPY="tma")
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-trimmed-channel-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=(D256Training if D==256 else WideTraining)(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];dt=p.dt.clone();owner=plan.b1 if D==256 else plan;oldln=owner.ln
 for rows,mb,mr,ar,reload in ((16,3,112,48,True),):
  op=TrimmedChannelLN(p,rows,mb,mr,ar,reload)
  print('LAUNCH',dict(rows=rows,minblocks=mb,registers=op.registers,requested=[mr,ar],cubin=str(op.cubin)),flush=True)
  def old():owner.ln=oldln;return plan()
  def new():owner.ln=op;return plan()
  yn,gn=new();torch.cuda.synchronize()
  v=dict(rows=rows,minblocks=mb,registers=[mr,ar],reload=reload,occupancy=op.occupancy,cubin=str(op.cubin),dt_error=error(p.dt,dt))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   v['times']=paired(dict(old_ln=oldln,new_ln=op,old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
