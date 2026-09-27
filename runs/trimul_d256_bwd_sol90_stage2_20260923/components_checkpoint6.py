"""Measure actual checkpoint6 components, diagnostic only."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint6 import Training
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,diagnostic_only=True)
path=THIS/f'components-checkpoint6-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();p=plan.p;b=plan.b1;o=plan.product_output;s=plan.schedule
 functions=dict(front=plan.f.front,
  output_norm=lambda:o.norm.launch((o.grid,1,1),(o.threads,1,1),[o.np],o.smem),
  output_epi=lambda:o.epi.launch((1056,1,1),(256,1,1),[o.ep],0),
  backward_epi=lambda:b.epi.launch((1056,1,1),(256,1,1),[b.ep],0),
  output_ln=plan.ln,source=plan.b7.source_only,
  prefix_copy=plan.dx.copy_prefix,weight_pack=plan.dx.pack_weights,input_ln_reduce=plan.dx.reduce_only)
 for name in s.args:
  def run(name=name):s.run(name)
  functions[name]=run
 if D==512:
  functions['backward_norm']=lambda:b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem)
  functions['source_gp']=plan.saved_source.derivatives
  functions['source_dw']=plan.saved_source.matmul
 if plan.split_dwp is not None:functions['split_dwp']=plan.split_dwp
 record['times']=paired(functions)
 print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
