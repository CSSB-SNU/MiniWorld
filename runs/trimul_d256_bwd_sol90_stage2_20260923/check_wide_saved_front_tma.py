from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint5 import Training
from wide_saved_front import SavedSource
from wide_saved_front_tma import SavedFrontTma as SavedFront
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-saved-front-tma-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone();part=p.floats[7].clone()
 original_front=plan.f.front;original_source=plan.b7.source_only
 pre=p.x.new_empty((D//8,p.M,64));front=SavedFront(original_front,pre);source=SavedSource(p,pre,plan.f.mask)
 def old():plan.f.front=original_front;plan.b7.source_only=original_source;return plan()
 def new():plan.f.front=front;plan.b7.source_only=source;return plan()
 pre.fill_(float('nan'));p.gp_all.fill_(float('nan'));p.floats[7][:,3*D*D:].fill_(float('nan'))
 yn,gn=new();torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['gp_error']=error(p.gp_all,gp);record['partial_error']=error(p.floats[7][:,3*D*D:],part[:,3*D*D:])
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 if record['strict']:
  wanted=source.partial.clone();candidates=[];functions={}
  for index in source.matmul.indices:
   source.matmul.index=index;source.partial.fill_(float('nan'));source.matmul();torch.cuda.synchronize()
   e=error(source.partial,wanted);choice=dict(index=index,error=e,algo=list(source.matmul.heuristics[index].algo.data))
   candidates.append(choice)
   if e==0:
    def run(index=index):source.matmul.index=index;source.matmul()
    functions[f'lt{index}']=run
  times=paired(functions)
  valid=[]
  for choice in candidates:
   if choice['error']==0:choice['time_us']=times[f'lt{choice["index"]}']['median_us'];valid.append(choice)
  best=min(valid,key=lambda c:c['time_us']);source.matmul.index=best['index']
  record['lt_candidates']=candidates;record['lt_selected']=best
  yn,gn=new();record['selected_errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  assert all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['selected_errors'].items())
 record['front_cubin']=str(front.cubin);record['source_cubin']=str(source.cubin);record['saved_bytes']=pre.numel()*2
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_front=original_front,new_front=front,old_source=original_source,new_source=source,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
