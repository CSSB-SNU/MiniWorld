from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint11 import Training
from wide_table_contract_gp import TableContractGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-table-contract-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();original=plan.contract_gp
 op=TableContractGP(plan)
 def old():plan.contract_gp=original;return plan()
 def new():plan.contract_gp=op;return plan()
 yn,gn=new();torch.cuda.synchronize()
 record['cubin']=str(op.cubin);record['gp_error']=error(p.gp_all,gp)
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
 record['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in record['errors'].items())
 print('CHECK',record,flush=True)
 if record['strict']:
  record['times']=paired(dict(old_gp=original,new_gp=op,old_full=old,new_full=new))
  print('TIMES',{k:t['median_us'] for k,t in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
