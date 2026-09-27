from pathlib import Path
import sys,os,json
from types import SimpleNamespace
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training
from d256_aliased_pipe_source import AliasedPipeSource
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
from d256_small_loader_source import SmallLoaderSource
import subprocess
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();p=plan.p
 launch=T._launch_module();tm=lambda t:launch.tensor_map(t,[64,32],dims=[p.M,2*D],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
 params=launch.Struct([p.maps[0],W.tm(p.w1),tm(p.dl),tm(p.dr),*[tm(t) for t in p.gp],plan.f.mask,p.floats[7],p.M])
 proxy=SimpleNamespace(params=params,splits=plan.b7.splits)
 alias=AliasedPipeSource(p,proxy,producer_regs=96);small=SmallLoaderSource(plan,n256=True)
 for label,cubin in [('current',plan.b7.cubin),('aliased',alias.cubin),('small_loader',small.cubin)]:
  print('RESOURCE',label,str(cubin),flush=True)
  subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
 for op in (plan.b7.source_only,alias,small):
  for _ in range(3):op()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
 plan.b7.source_only();alias();small()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
