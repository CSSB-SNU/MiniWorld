"""Measure actual checkpoint9 components, diagnostic only."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,diagnostic_only=True)
path=THIS/f'components-d256-gate-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();p=plan.p;b=plan.b1;o=plan.f.output;s=plan.schedule
 functions=dict(front=plan.f.front,
  output_norm=lambda:o.norm.launch((o.grid,1,1),(o.threads,1,1),[o.np],o.smem),
  output_epi=lambda:o.epi.launch((1056,1,1),(256,1,1),[o.ep],0),
  backward_epi=b.prepare,
  output_ln=b.ln,source=plan.b7.source_only,
  prefix_copy=plan.dx.copy_prefix,weight_pack=plan.dx.pack_weights,input_ln_reduce=plan.dx.reduce_only)
 for name in s.args:
  def run(name=name):s.run(name)
  functions[name]=run
 functions['dx_lt']=plan.dx_lt
 record['times']=paired(functions)
 print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
