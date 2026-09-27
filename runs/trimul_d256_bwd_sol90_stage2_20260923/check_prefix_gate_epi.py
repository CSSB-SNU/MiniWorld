from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint8 import Training as WideTraining
from d256_lt_checkpoint import Training as D256Training
from prefix_gate_epi import PrefixGateEpi
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-prefix-gate-epi-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=(D256Training if D==256 else WideTraining)(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    dp=p.tensors[7].clone();dg=p.dg.clone();wantdw=p.dwg.clone()
    op=PrefixGateEpi(plan);op();torch.cuda.synchronize()
    record.update(dp_error=error(p.tensors[7],dp),dg_error=error(op.prefix.t(),dg),cubin=str(op.cubin))
    workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    dw=LtBmm(op.prefix.unsqueeze(0),p.xn.reshape(p.M,D).unsqueeze(0),p.dwg.unsqueeze(0),workspace)
    choices=[];functions={}
    for index in dw.indices:
        dw.index=index;dw();torch.cuda.synchronize();err=error(p.dwg,wantdw)
        choice=dict(index=index,error=err,algo=list(dw.heuristics[index].algo.data));choices.append(choice)
        if err==0:
            def run(index=index):dw.index=index;dw()
            functions[f'lt{index}']=run
    record['choices']=choices;print('INPUT', {k:v for k,v in record.items() if k!='choices'},flush=True)
    if functions:
        times=paired(functions)
        choice=min((c for c in choices if c['error']==0),key=lambda c:times[f'lt{c["index"]}']['median_us'])
        dw.index=choice['index'];record['selected']=choice
        oldrun=plan.schedule.run;oldcopy=plan.dx.copy_prefix
        oldepi=plan.b1.prepare if D==256 else plan.b1.epi
        def newrun(name):
            if name=='dwg':dw()
            else:oldrun(name)
        def baseline():
            plan.schedule.run=oldrun;plan.dx.copy_prefix=oldcopy
            if D==256:plan.b1.prepare=oldepi
            else:plan.b1.epi=oldepi
            return plan()
        def candidate():
            plan.schedule.run=newrun;plan.dx.copy_prefix=lambda:None
            if D==256:plan.b1.prepare=op
            else:plan.b1.epi=op
            return plan()
        yn,gn=candidate();torch.cuda.synchronize()
        record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
        print('CHECK',record['strict'],record['errors'],flush=True)
        def oldepi_copy():
            if D==256:oldepi()
            else:oldepi.launch((1056,1,1),(256,1,1),[plan.b1.ep],0)
            oldcopy()
        if record['strict']:
            record['times']=paired(dict(baseline_epi_copy=oldepi_copy,new_epi_copy=op,baseline_full=baseline,new_full=candidate))
            print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
