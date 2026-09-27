from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from wide_dense_output import DenseOutput
from wide_tile_ln import TileLN
os.environ.update(SHARED_CANDIDATE='saved_products',SAVE_EPI_GRID='1056',SAVE_EPI_THREADS='256',
 GP_OFF='1',GP_WARP='1',DX_LN='1',DX_WARP='1',SAVE_NORM='1',DX_N256='0',DX_PIPE='0',DX_SPLIT='0',
 DX_IN_STATS='0',DX_GAMMA_CACHE='0',LN_THREADS='128',LN_MINBLOCKS='3',LN_AGG='1',LN_CACHE='0',
 LN_STATS_TMA='0',LN_STORE_READ='0',LN_GAMMA_SMEM='0',DN_SLIM='0')
N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(D=256,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-output-schedule-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);attach(plan);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();old_out=plan.f.output;old_ln=plan.b1.ln
 products=plan.b1.prepare.original
 outs={0:old_out,128:DenseOutput(plan.f,p,(products.proj,products.gate),128),256:DenseOutput(plan.f,p,(products.proj,products.gate),256)}
 lns={0:old_ln,16:TileLN(p,16,True,False),32:TileLN(p,32,True,False)}
 for nt,rows in ((0,16),(0,32),(128,0),(256,0),(128,16),(128,32)):
  def old():plan.f.output=old_out;plan.b1.ln=old_ln;return plan()
  def new():plan.f.output=outs[nt];plan.b1.ln=lns[rows];return plan()
  yn,gn=new();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row=dict(output_threads=nt,ln_rows=rows,errors=es)
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_full=old,new_full=new))
   print('TIMES',nt,rows,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
