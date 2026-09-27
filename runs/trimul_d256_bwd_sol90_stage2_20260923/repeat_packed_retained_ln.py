"""Test current small LN tiles with direct shared layout or local transpose barriers."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
from wide_cached_ln import CachedLN
from wide_prefetch_ln import PrefetchLN
from wide_packed_retained_ln import PackedRetainedLN
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index];N=384;mode=0
if D==256:from d256_permuted_ln_checkpoint import Training
else:from wide_checkpoint19 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-repeat-packed-retained-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone()
    owner=plan.b1 if D==256 else plan;oldln=owner.ln
    rows=32 if D==256 and mode==0 else 16;minblocks=(3 if D==256 else 2) if mode==0 else (4 if D==256 else 3)
    base=(PrefetchLN if D==384 and mode==0 else CachedLN)(p,rows,True,False,minblocks)
    for candidate in range(3):
        op=PackedRetainedLN(p,base,rows,minblocks)
        item=dict(round=candidate,mode=mode,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy)
        record['candidates'].append(item)
        def before():owner.ln=oldln;return plan()
        def after():owner.ln=op;return plan()
        yn,gn=after();torch.cuda.synchronize();item['dt_error']=error(p.dt,dt)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict'] and op.local_bytes==0:
            item['stage_times']=paired(dict(old_ln=oldln,new_ln=op));gc.collect()
            def oldbw():owner.ln=oldln;return plan.backward()
            def newbw():owner.ln=op;return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
