from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from wide_stream_contract_gp import StreamContractGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-stream-contract-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    old=plan.contract_gp;groups=int(os.environ.get('GP_GROUPS','2'));read_credit=os.environ.get('GP_READ_CREDIT','0')=='1';op=StreamContractGP(plan,groups=groups,read_credit=read_credit)
    for grid in (132,264):
        op.grid=grid;item=dict(grid=grid,groups=groups,read_credit=read_credit,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes)
        record['candidates'].append(item);print('LAUNCH',item,flush=True)
        p.gp_all.fill_(float('nan'));op();torch.cuda.synchronize();item['gp_error']=error(p.gp_all,gp)
        def before():plan.contract_gp=old;return plan()
        def after():plan.contract_gp=op;return plan()
        yn,gn=after();torch.cuda.synchronize()
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=item['gp_error']==0 and all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['stage_times']=paired(dict(old_gp=old,new_gp=op));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in item[key].items()},flush=True)
        before();path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
