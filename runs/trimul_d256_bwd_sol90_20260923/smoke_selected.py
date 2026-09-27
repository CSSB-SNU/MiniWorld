from pathlib import Path
import sys,os,json
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent.parent/'.engine-release-2.0.0/src'))
sys.path.insert(0,str(R.parent/'trimul_cuda_widths_opt_20260923'))
sys.path.insert(0,str(R.parent/'trimul_forward_wide_20260923'))
from fixture import setup
from validate_engine import error,capture
sys.path.insert(0,str(R))
from selected import Training
import torch
N=int(os.environ.get('LENGTH','384'))
leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad():
 p=Training(leaves,mask,ds,dy)
 y,grad=p();expected=[x.clone() for x in (y,*grad)]
 g,_=capture(p)
 leaves[1].mul_(.9);dy.mul_(.8)
 y,grad=p();expected=[x.clone() for x in (y,*grad)]
 g.replay();torch.cuda.synchronize()
 errors=[error(x,z) for x,z in zip((p.f.output.y,*p.p.outputs),expected)]
 assert max(errors)<5e-6,errors
 assert p.p.w1.data_ptr()==p.f.w.data_ptr()
 (R/f'selected-L{N}.json').write_text(json.dumps(dict(errors=errors,live_pack_shared=True),indent=2))
 print('SELECTED_PASS',N,max(errors),flush=True)
