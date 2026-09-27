"""Profile output LN with its actual backward producer context, same buffers."""
from pathlib import Path
import sys,os
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint21 import Training
from safe_early_affine_checkpoint import attach
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,*_=setup(512,384)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy)
    old=(plan.prefix_gate,plan.ln,plan.dx.reduce_only)
    attach(plan)
    new=(plan.prefix_gate,plan.ln,plan.dx.reduce_only)
    def select(items):
        plan.prefix_gate,plan.ln,plan.dx.reduce_only=items
        plan.b1.epi=plan.prefix_gate
    for items in (old,new):
        select(items)
        for _ in range(3):plan()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name,items in (('checkpoint21',old),('ordered_pair_early',new)):
        select(items)
        print('CONTEXT',name,flush=True)
        torch.cuda.nvtx.range_push(name)
        plan.backward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
