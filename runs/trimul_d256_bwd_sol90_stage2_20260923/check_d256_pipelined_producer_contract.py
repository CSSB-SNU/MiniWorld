from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from d256_pipelined_producer_contract import PipelinedProducerContract
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));consumer=(224,232)[index]
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
record=dict(D=256,L=384,consumer=consumer,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-pipelined-producer-contract-L384-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];deriv=[p.dl.clone(),p.dr.clone()]
    op=PipelinedProducerContract(plan,consumer);oldrun=plan.schedule.run
    p.dl.fill_(float('nan'));p.dr.fill_(float('nan'));op();torch.cuda.synchronize()
    record['pool_metadata']=op.pool_metadata
    record.update(dl_error=error(p.dl,deriv[0]),dr_error=error(p.dr,deriv[1]),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,cubin=str(op.cubin))
    def run(name):
        if name=='bc0':op()
        elif name not in ('bc1','bc2','bc3'):oldrun(name)
    def before():plan.schedule.run=oldrun;return plan()
    def after():plan.schedule.run=run;return plan()
    yn,gn=after();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        def oldcontract():
            for name in ('bc0','bc1','bc2','bc3'):oldrun(name)
        record['stage_times']=paired(dict(old_contract=oldcontract,new_contract=op));gc.collect()
        def oldbw():plan.schedule.run=oldrun;return plan.backward()
        def newbw():plan.schedule.run=run;return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['stage_times','backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
