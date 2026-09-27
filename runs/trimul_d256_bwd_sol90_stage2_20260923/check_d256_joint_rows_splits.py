"""Validate outstanding dW/next-preactivation overlap without reassociation."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_joint_rows_source import JointRowsSource
from wide_cached_input import CachedInput
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-joint-rows-splits-source-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    part=p.floats[7].reshape(-1)[3*D*D:].as_strided((plan.b7.splits,8*D*D),(11*D*D,1));partial=part.clone()
    source=plan.b7.source_only;old_reduce=plan.dx.reduce_only
    for splits in (4,16):
        readwait=160
        op=JointRowsSource(plan,readwait,splits)
        reduce=CachedInput(p,16,128,4,splits=splits)
        item=dict(splits=splits,consumer_regs=readwait,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy)
        print('RESOURCE',item,flush=True)
        record['candidates'].append(item)
        path.write_text(json.dumps(record,indent=2))
        if op.registers*3 < 2*readwait+32:
            item['register_pool_insufficient']=True
            continue
        def before():plan.b7.source_only=source;plan.dx.reduce_only=old_reduce;return plan()
        def after():plan.b7.source_only=op;plan.dx.reduce_only=reduce;return plan()
        p.gp_all.fill_(float('nan'));part.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
        item['gp_error']=error(p.gp_all,gp)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict'] and not op.local_bytes:
            item['stage_times']=paired(dict(old_source=source,new_source=op));gc.collect()
            def oldbw():plan.b7.source_only=source;plan.dx.reduce_only=old_reduce;return plan.backward()
            def newbw():plan.b7.source_only=op;plan.dx.reduce_only=reduce;return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
