from pathlib import Path
import sys,os
from types import SimpleNamespace
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint7 import Training
from wide_two_group_contract_gp import TwoGroupContractGP
from wide_saved_front_transposed import SavedTransposedFront
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=512;N=int(os.environ.get('LENGTH','768'))
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();p=plan.p
    pre=p.x.new_empty((8*D,p.M));front=SavedTransposedFront(plan.baseline_front,pre);front()
    op=TwoGroupContractGP(SimpleNamespace(p=p,pre=pre,f=plan.f))
    for _ in range(3):op()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    op();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
