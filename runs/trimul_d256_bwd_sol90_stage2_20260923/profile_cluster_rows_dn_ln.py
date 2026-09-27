from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint19 import Training
from cluster_rows_dn_ln_timed import TimedClusterRowsDnLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=512;N=384
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();op=TimedClusterRowsDnLN(plan)
    for _ in range(3):op()
    torch.cuda.synchronize();ticks=op.ticks.cpu()
    phases=('gemm','gemm_cluster_wait','exchange','ln_and_store','affine_aggregate_wait','affine_remote','final_wait')
    record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),cubin=str(op.cubin),phases_cycles={})
    for i,name in enumerate(phases):
        t=(ticks[:,i+1]-ticks[:,i]).double()
        record['phases_cycles'][name]=dict(median=t.median().item(),p90=t.quantile(.9).item(),maximum=t.max().item())
    print('PHASES',record,flush=True)
    (THIS/f'profile-cluster-rows-D{D}-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
