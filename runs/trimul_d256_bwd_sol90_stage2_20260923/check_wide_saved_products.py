from pathlib import Path
import sys,os,json,ctypes
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
from wide_saved_products import SavedWideProductsOutput
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-saved-products-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p;b1=plan.b1
 output=SavedWideProductsOutput(plan.f,p,products=(b1.proj,b1.gate))
 def bwd():
  if D==512:
   b1.norm.launch((b1.grid,1,1),(b1.threads,1,1),[b1.np],b1.smem)
   torch.mm(p.tensors[6],b1.wp.t(),out=b1.proj)
  b1.epi.launch((1056,1,1),(256,1,1),[b1.ep],0)
  torch.mm(p.tensors[7],b1.wp,out=p.tensors[9])
  if b1.exact_dwp is None:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
  else:b1.exact_dwp()
  torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg)
  L=T._launch_module();drv=b1.ln.unit.drv;args=L._Packed([b1.lp])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(b1.ln.handle)),b1.grid,1,1,b1.threads,1,1,b1.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
  plan.contract();plan.b7.source_only();plan.dx();return p.outputs
 def forward():
  plan.f.mask.copy_(mask);p.mask.copy_(mask.reshape_as(p.mask));plan.f.output=output;return plan.f()
 def new():return forward(),bwd()
 y,g=plan();oldy=y.clone();expected=[t.clone() for t in g];proj=b1.proj.clone();gate=b1.gate.clone()
 b1.proj.fill_(float('nan'));b1.gate.fill_(float('nan'));forward()
 record['saved_gate_error']=error(b1.gate,gate)
 if D==384:record['saved_projection_error']=error(b1.proj,proj)
 yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubin']=str(output.cubin);print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_full=plan,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
