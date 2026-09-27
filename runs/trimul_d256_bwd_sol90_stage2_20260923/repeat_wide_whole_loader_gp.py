from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint23 import Training
from wide_loader_warp_gp import LoaderWarpGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//3];unroll=False;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,unroll=unroll,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'repeat-wide-whole-loader-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();old=plan.contract_gp
    op=LoaderWarpGP(plan,unroll,True)
    record.update(registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,cubin=str(op.cubin))
    print('RESOURCE',record,flush=True);path.write_text(json.dumps(record,indent=2))
    def before():plan.contract_gp=old;return plan()
    def after():plan.contract_gp=op;return plan()
    p.gp_all.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
    record['gp_error']=error(p.gp_all,gp)
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        record['stage_times']=paired(dict(old_contract=old,new_contract=op));gc.collect()
        def oldbw():plan.contract_gp=old;return plan.backward()
        def newbw():plan.contract_gp=op;return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['stage_times','backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
