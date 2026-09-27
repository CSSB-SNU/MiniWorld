from pathlib import Path
import sys,os
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=512;N=768
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();op=plan.ln
    for _ in range(3):op()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    op();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
