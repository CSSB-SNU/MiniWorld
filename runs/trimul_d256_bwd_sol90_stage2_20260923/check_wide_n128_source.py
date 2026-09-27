from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint2 import Training
from wide_offset_source import OffsetSource as PipeSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
os.environ['WIDE_SOURCE_N128']='1'
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),offsets=os.environ.get('WIDE_PIPE_OFF')=='1',complete=False)
path=THIS/f'result-wide-n128-source-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone();part=p.floats[7].clone()
 candidate=PipeSource(p,plan.b7);oldsource=plan.b7.source_only
 p.gp_all.fill_(float('nan'));p.floats[7][:,3*D*D:].fill_(float('nan'))
 candidate();torch.cuda.synchronize()
 record['derivative_error']=error(p.gp_all,gp);record['partial_error']=error(p.floats[7][:,3*D*D:],part[:,3*D*D:])
 assert record['derivative_error']==0 and record['partial_error']==0,record
 def old():plan.b7.source_only=oldsource;return plan()
 def new():plan.b7.source_only=candidate;return plan()
 yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubin']=str(candidate.cubin);print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_source=oldsource,new_source=candidate,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
