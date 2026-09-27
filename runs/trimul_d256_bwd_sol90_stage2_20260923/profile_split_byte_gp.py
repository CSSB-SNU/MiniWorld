"""Diagnose actual DRAM traffic; profiling is not paired performance evidence."""
from pathlib import Path
import sys,os
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint19 import Training
from wide_split_byte_gp import SplitBytePre,SplitByteGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,*_=setup(512,384)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan()
    owners=[];ops=[plan.contract_gp]
    for xor in (0,127):
        packed=SplitBytePre(plan.pre,xor);packed();op=SplitByteGP(plan,packed)
        owners.append(packed);ops.append(op)
    for _ in range(3):
        for op in ops:op()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for op in ops:op()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
