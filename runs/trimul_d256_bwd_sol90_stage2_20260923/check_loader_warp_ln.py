"""Pilot output LN with one independent TMA issuer warp."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_loader_warp_ln import LoaderWarpLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=(256,384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=384
if D==256:from d256_permuted_ln_checkpoint import Training
else:from wide_checkpoint19 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-loader-warp-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone()
    owner=plan.b1 if D==256 else plan;old=owner.ln
    choices=((16,3),(16,2)) if D==256 else ((16,2),(8,3)) if D==384 else ((8,2),)
    for rows,mb in choices:
        op=LoaderWarpLN(p,rows,mb)
        item=dict(rows=rows,minblocks=mb,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy)
        record['candidates'].append(item)
        def before():owner.ln=old;return plan()
        def after():owner.ln=op;return plan()
        yn,gn=after();torch.cuda.synchronize();item['dt_error']=error(p.dt,dt)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict'] and not op.local_bytes:
            item['stage_times']=paired(dict(old_ln=old,new_ln=op));gc.collect()
            def oldbw():owner.ln=old;return plan.backward()
            def newbw():owner.ln=op;return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
