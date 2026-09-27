from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from split_weight_product import SplitWeightProduct
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
if D==256:from d256_lt_checkpoint import Training
else:from wide_checkpoint7 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,operations={})
path=THIS/f'result-split-weight-product-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();oldy=y.clone();expected=[t.clone() for t in g]
 original_run=plan.schedule.run;original_split=getattr(plan,'split_dwp',None);selected={}
 for name,a,b,out in (('dwp',p.tensors[7],p.tensors[6],p.dwp),('dwg',p.dg,p.xn.reshape(p.M,D),p.dwg)):
  wanted=out.clone();rows=[];ops={};functions={}
  def control(name=name):
   if name=='dwp' and original_split is not None:original_split()
   else:original_run(name)
  functions['old']=control
  for splits in (4,8,16,32):
   op=SplitWeightProduct(a,b,out,splits);ops[splits]=op
   for i in op.matmul.indices:
    op.matmul.index=i;op();torch.cuda.synchronize();err=error(out,wanted)
    key=f's{splits}i{i}';row=dict(key=key,splits=splits,index=i,error=err,algo=list(op.matmul.heuristics[i].algo.data),cubin=str(op.cubin));rows.append(row)
    if err<4e-4:
     def run(op=op,i=i):op.matmul.index=i;op()
     functions[key]=run
  times=paired(functions)
  for row in rows:
   if row['key'] in times:row['time_us']=times[row['key']]['median_us']
  valid=[v for v in rows if 'time_us' in v]
  choice=min(valid,key=lambda v:v['time_us']) if valid else None
  if choice and choice['time_us']<times['old']['median_us']*.98:
   op=ops[choice['splits']];op.matmul.index=choice['index'];selected[name]=op
  else:choice=None
  record['operations'][name]=dict(candidates=rows,old_us=times['old']['median_us'],selected=choice)
  print('SELECT',name,record['operations'][name]['old_us'],choice,flush=True);path.write_text(json.dumps(record,indent=2))
 def run(name):
  if name in selected:selected[name]()
  else:original_run(name)
 def old():
  plan.schedule.run=original_run
  if original_split is not None:plan.split_dwp=original_split
  return plan()
 def new():
  plan.schedule.run=run
  if original_split is not None:plan.split_dwp=selected.get('dwp',original_split)
  return plan()
 yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('CHECK',record['strict'],record['errors'],flush=True)
 if record['strict']:
  record['times']=paired(dict(old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
