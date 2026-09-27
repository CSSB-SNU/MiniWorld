"""Exact contraction coverage and graph fork/join for chunked output LN."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_early_affine_checkpoint import Training
from d256_chunked_ln_contract import ChunkedLNContract
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));overlap=bool(index);D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,overlap=overlap,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-chunked-ln-contract-L384-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    intermediates=[p.dt.clone(),p.dl.clone(),p.dr.clone()]
    op=ChunkedLNContract(plan,overlap)
    def before():op.select(False);return plan()
    def after():op.select(True);return plan()
    for t in (p.dt,p.dl,p.dr,*p.floats[8:12]):t.fill_(float('nan'))
    yn,gn=after();torch.cuda.synchronize();actual=[yn.clone(),*[v.clone() for v in gn]]
    record.update(resources=op.resources,cubins=[str(op.ln_cubin),str(op.contract_cubin)])
    record['intermediate_errors']={n:error(a,b) for n,a,b in zip(('dt','dl','dr'),(p.dt,p.dl,p.dr),intermediates)}
    record['errors']={n:error(a,b) for n,a,b in zip(names,actual,expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not any(v['local_bytes'] for v in op.resources.values()):
        graph,out=capture(after)
        for t in (p.dt,p.dl,p.dr,*p.floats[8:12]):t.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],actual)}
        assert max(record['graph_errors'].values())<5e-6,record['graph_errors']
        del graph,out;gc.collect()
        def oldbw():op.select(False);return plan.backward()
        def newbw():op.select(True);return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
