from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_dense_checkpoint import Training
from d256_mask_source import MaskSource
from prefix_dx import PrefixDx
from wide_cached_input import CachedInput
from saved_input_stats import enable as enable_input
from tma_b7 import TmaB7
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-prefix-cached-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p
 finish=PrefixDx(p,leaves,splits=8 if N==384 else 16)
 plan.b7=TmaB7(p,leaves,packed=True,mask=plan.f.mask)
 enable_input(plan,gamma_cache=True)
 plan.b7=MaskSource(plan.b7,'bulk');original_finish=plan.b7.wide_finish
 finish.reduce_only=CachedInput(p,16,128,4,splits=plan.b7.splits)
 y,g=plan();oldy=y.clone();expected=[t.clone() for t in g]
 finish.matrix_product();wanted=p.tensors[10].clone()
 workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
 lt=LtBmm(finish.input.t().unsqueeze(0),finish.weights.unsqueeze(0),p.tensors[10].unsqueeze(0),workspace)
 choices=[];functions={}
 for index in lt.indices:
  lt.index=index;lt.out.fill_(float('nan'));lt();torch.cuda.synchronize();err=error(lt.out,wanted)
  choice=dict(index=index,error=err,algo=list(lt.heuristics[index].algo.data));choices.append(choice)
  if err==0:
   def run(index=index):lt.index=index;lt()
   functions[f'lt{index}']=run
 times=paired(functions)
 for c in choices:
  if c['error']==0:c['time_us']=times[f'lt{c["index"]}']['median_us']
 best=min((c for c in choices if c['error']==0),key=lambda c:c['time_us']);lt.index=best['index']
 record['lt_candidates']=choices;record['selected']=best
 finish.gemm_only=lt
 def old():plan.b7.wide_finish=original_finish;return plan()
 def new():plan.b7.wide_finish=finish;return plan()
 yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_dx=original_finish,new_dx=finish,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
