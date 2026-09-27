"""Apply the latest wide saved-front/fused-contraction design to D256."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_stats_checkpoint import Training
from d256_saved_stats_overlap_front import SavedStatsOverlapFront
from wide_two_group_contract_gp import TwoGroupContractGP
from wide_staged_epilogue_gp import StagedEpilogueGP
from wide_cached_input import CachedInput
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-saved-fused-contract-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    front=plan.f.front;source=plan.b7.source_only;run=plan.schedule.run;reduce=plan.dx.reduce_only
    plan.pre=p.x.new_empty((8*D,p.M));newfront=SavedStatsOverlapFront(front,plan.pre)
    plan.contract_gp=TwoGroupContractGP(plan,channel_group=1)
    plan.contract_gp=StagedEpilogueGP(plan)
    ws=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    for splits in (8,16,32):
        step=p.M//splits
        partial=p.floats[7].reshape(-1)[3*D*D:].as_strided((splits,8*D,D),(11*D*D,D,1))
        a=p.gp_all.as_strided((splits,8*D,step),(step,p.M,1))
        b=p.xn.as_strided((splits,step,D),(step*D,D,1))
        dw=LtBmm(a,b,partial,ws)
        finish=CachedInput(p,16,128,4,splits=splits)
        item=dict(splits=splits,front=str(newfront.cubin),gp=str(plan.contract_gp.cubin),dw_algo=list(dw.heuristics[dw.index].algo.data))
        record['candidates'].append(item)
        def newsource():plan.contract_gp();dw()
        def newrun(name):
            if name not in ('bc0','bc1','bc2','bc3'):run(name)
        def select(new):
            plan.f.front=newfront if new else front
            plan.b7.source_only=newsource if new else source
            plan.schedule.run=newrun if new else run
            plan.dx.reduce_only=finish if new else reduce
        def before():select(False);return plan()
        def after():select(True);return plan()
        plan.pre.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
        item['gp_error']=error(p.gp_all,gp)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            def oldsource():
                for name in ('bc0','bc1','bc2','bc3'):run(name)
                source()
            item['source_times']=paired(dict(old_source=oldsource,new_source=newsource));gc.collect()
            def oldbw():select(False);return plan.backward()
            def newbw():select(True);return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('source_times','backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
