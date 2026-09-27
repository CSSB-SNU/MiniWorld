"""Current qualified short paths: profiler trace or separate graph diagnostics."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
from d256_pool_checkpoint import Training
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy)
    for _ in range(3):plan()
    torch.cuda.synchronize()
    if os.environ.get('MEASURE_COMPONENTS')!='1':
        torch.cuda.cudart().cudaProfilerStart();plan.backward();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
