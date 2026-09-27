"""Include both half-save forward cost and half-recompute backward in timing."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_saved_half_front import SavedHalfFront
from d256_saved_half_source import SavedHalfSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-saved-half-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    part=p.floats[7].reshape(-1)[3*D*D:].as_strided((plan.b7.splits,8*D*D),(11*D*D,1));partial=part.clone()
    front=plan.f.front;source=plan.b7.source_only
    pre=p.x.new_empty((4*D,p.M))
    for half in (1,0):
        newfront=SavedHalfFront(front,pre,half);op=SavedHalfSource(plan,pre,half)
        item=dict(saved_half='projection' if half else 'gate',saved_bytes=pre.numel()*pre.element_size(),front=str(newfront.cubin),source=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,front_smem=newfront.smem,source_smem=op.smem)
        record['candidates'].append(item)
        def select(new):plan.f.front=newfront if new else front;plan.b7.source_only=op if new else source
        def before():select(False);return plan()
        def after():select(True);return plan()
        pre.fill_(float('nan'));p.gp_all.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
        item['gp_error']=error(p.gp_all,gp);item['partial_error']=error(part,partial)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['stage_times']=paired(dict(old_source=source,new_source=op));gc.collect()
            def oldbw():select(False);return plan.backward()
            def newbw():select(True);return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
