from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_two_cta_prefix_dx import TwoCtaPrefixDX
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384;slots,depth=((2,0),(3,0),(3,1))[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,slots=slots,depth=depth,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-two-cta-prefix-dx-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dxn=p.tensors[10].clone()
    op=TwoCtaPrefixDX(plan,slots,depth);old=plan.dx.gemm_only
    p.tensors[10].fill_(float('nan'));op();torch.cuda.synchronize()
    record.update(dxn_error=error(p.tensors[10],dxn),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,cubin=str(op.cubin))
    def before():plan.dx.gemm_only=old;return plan()
    def after():plan.dx.gemm_only=op;return plan()
    yn,gn=after();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        record['stage_times']=paired(dict(old_dx=old,new_dx=op));gc.collect()
        def oldbw():plan.dx.gemm_only=old;return plan.backward()
        def newbw():plan.dx.gemm_only=op;return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['stage_times','backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
