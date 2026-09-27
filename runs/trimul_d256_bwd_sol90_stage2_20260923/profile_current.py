from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from selected_current import Training,Previous
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy)
 from shared_candidate import attach
 attach(plan)
 if os.environ.get('PROFILE_CLUSTER')=='1':
  from cluster_b7 import ClusterB7
  plan.b7=ClusterB7(plan.p,plan.f.mask)
 if os.environ.get('PROFILE_RING3')=='1':
  from ring3 import Ring3
  plan.b7=Ring3(plan.p,leaves)
 for _ in range(3):plan()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();plan.backward();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
