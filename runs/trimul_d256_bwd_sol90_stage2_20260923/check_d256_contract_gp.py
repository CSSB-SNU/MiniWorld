from pathlib import Path
import sys,os,json
from types import SimpleNamespace
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_lt_checkpoint import Training
from wide_two_group_contract_gp import TwoGroupContractGP
from wide_saved_front_transposed import SavedTransposedFront
from wide_cached_input import CachedInput
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-contract-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    pre=p.x.new_empty((8*D,p.M));oldfront=plan.f.front
    front=SavedTransposedFront(oldfront,pre);front();torch.cuda.synchronize()
    fused=TwoGroupContractGP(SimpleNamespace(p=p,pre=pre,f=plan.f),channel_group=2);fused();torch.cuda.synchronize()
    record['gp_error']=error(p.gp_all,gp)
    splits=8 if N==384 else 16;step=p.M//splits
    partial=p.floats[7].reshape(-1)[3*D*D:].as_strided((splits,8*D,D),(11*D*D,D,1))
    a=p.gp_all.as_strided((splits,8*D,step),(step,p.M,1));b=p.xn.as_strided((splits,step,D),(step*D,D,1))
    workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8);dw=LtBmm(a,b,partial,workspace)
    reduce=CachedInput(p,16,128,4,splits=splits)
    dw();torch.cuda.synchronize();wanted=partial.clone();choices=[];functions={}
    for index in dw.indices:
        dw.index=index;dw();torch.cuda.synchronize();err=error(partial,wanted)
        choice=dict(index=index,error=err,algo=list(dw.heuristics[index].algo.data));choices.append(choice)
        if err==0:
            def run(index=index):dw.index=index;dw()
            functions[f'lt{index}']=run
    times=paired(functions)
    choice=min((c for c in choices if c['error']==0),key=lambda c:times[f'lt{c["index"]}']['median_us'])
    dw.index=choice['index'];record['dw_selected']=choice;record['dw_candidates']=choices;record['splits']=splits
    oldrun=plan.schedule.run;oldsource=plan.b7.source_only;oldreduce=plan.dx.reduce_only
    def fusedrun(name):
        if name not in ('bc0','bc1','bc2','bc3'):oldrun(name)
    def fusedsource():fused();dw()
    def baseline():
        plan.f.front=oldfront;plan.schedule.run=oldrun;plan.b7.source_only=oldsource;plan.dx.reduce_only=oldreduce;return plan()
    def candidate():
        plan.f.front=front;plan.schedule.run=fusedrun;plan.b7.source_only=fusedsource;plan.dx.reduce_only=reduce;return plan()
    yn,gn=candidate();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    record['cubins']=[str(front.cubin),str(fused.cubin),str(reduce.cubin)]
    print('CHECK',{k:v for k,v in record.items() if k!='dw_candidates'},flush=True);path.write_text(json.dumps(record,indent=2))
    def oldboth():
        for name in ('bc0','bc1','bc2','bc3'):oldrun(name)
        oldsource()
    if record['strict']:
        record['times']=paired(dict(baseline_front=oldfront,new_front=front,baseline_contract_source=oldboth,new_contract_source=fusedsource,baseline_full=baseline,new_full=candidate))
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
