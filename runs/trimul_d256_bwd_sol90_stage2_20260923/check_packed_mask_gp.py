from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint15 import Training
from wide_packed_mask_gp import PackedMaskGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,cases=[])
path=THIS/f'result-packed-mask-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;plan();old=plan.contract_gp;op=PackedMaskGP(plan);oldfront=plan.f.front
    def front():oldfront();op.encode()
    def before():plan.contract_gp=old;plan.f.front=oldfront;return plan()
    def after():plan.contract_gp=op;plan.f.front=front;return plan()
    original_mask=mask.clone()
    for case in ('binary','nonbinary'):
        if case=='nonbinary':
            mask.view(-1)[::7]=.375;mask.view(-1)[1::11]=-.25;mask.view(-1)[2::13]=-0.0
        y,g=before();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
        p.gp_all.fill_(float('nan'));op.packed.fill_(-1);yn,gn=after();torch.cuda.synchronize()
        item=dict(case=case,gp_error=error(p.gp_all,gp));record['cases'].append(item)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
    mask.copy_(original_mask);after();record['strict']=all(c['strict'] for c in record['cases'])
    record.update(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes)
    if record['strict']:
        record['stage_times']=paired(dict(old_gp=old,new_gp=op));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
