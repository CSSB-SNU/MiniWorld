from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint9 import Training as WideTraining
from d256_gate_checkpoint import Training as D256Training
from prefix_gate_dn_two import PrefixGateDnTwo as PrefixGateDn
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-prefix-gate-dn-two-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=(D256Training if D==256 else WideTraining)(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]]
    dn=p.tensors[9].clone();dp=p.tensors[7].clone();dg=plan.dx.input[:D].clone()
    op=PrefixGateDn(plan);op();torch.cuda.synchronize()
    record.update(dn_error=error(p.tensors[9],dn),dp_error=error(p.tensors[7],dp),dg_error=error(plan.dx.input[:D],dg),cubin=str(op.cubin))
    oldepi=plan.b1.prepare if D==256 else plan.b1.epi;oldrun=plan.schedule.run
    def newrun(name):
        if name!='dn':oldrun(name)
    def old():
        plan.schedule.run=oldrun
        if D==256:plan.b1.prepare=oldepi
        else:plan.b1.epi=oldepi
        return plan()
    def new():
        plan.schedule.run=newrun
        if D==256:plan.b1.prepare=op
        else:plan.b1.epi=op
        return plan()
    yn,gn=new();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True)
    def separate():oldepi();oldrun('dn')
    if record['strict']:
        record['times']=paired(dict(old_epi_dn=separate,new_epi_dn=op,old_full=old,new_full=new))
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
