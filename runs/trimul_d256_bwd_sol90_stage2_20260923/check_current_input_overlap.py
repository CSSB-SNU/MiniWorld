"""Overlap the current fused-contraction consumers using explicit graph events."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint17 import Training
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-current-input-overlap-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    oldsource=plan.b7.source_only;oldrun=plan.schedule.run
    assert plan.input_dw.workspace.data_ptr()!=plan.schedule.workspace.data_ptr()
    side=torch.cuda.Stream(device=p.x.device);ready=torch.cuda.Event();done=torch.cuda.Event()
    for mode in ('dw_first','dx_first'):
        def run(name):
            if name!='dx':return oldrun(name)
            main=torch.cuda.current_stream();ready.record(main)
            with torch.cuda.stream(side):
                side.wait_event(ready)
                if mode=='dw_first':plan.input_dw()
                else:oldrun('dx')
                done.record(side)
            if mode=='dw_first':oldrun('dx')
            else:plan.input_dw()
            # Input-LN also reduces all dW partials, so both must complete.
            main.wait_event(done)
        def install(new):
            plan.b7.source_only=plan.contract_gp if new else oldsource
            plan.schedule.run=run if new else oldrun
        def before():install(False);return plan()
        def after():install(True);return plan()
        yn,gn=after();torch.cuda.synchronize()
        item=dict(mode=mode,errors={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)})
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        record['candidates'].append(item);print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            actual=[x.clone() for x in [yn,*gn]];graph,out=capture(after);graph.replay();torch.cuda.synchronize()
            item['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],actual)}
            assert max(item['graph_errors'].values())<5e-6,item
            del graph,out;gc.collect()
            def oldback():install(False);return plan.backward()
            def newback():install(True);return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldback,new_backward=newback));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',mode,{k:v['median_us'] for key in ('backward_times','full_times') for k,v in item[key].items()},flush=True)
        before();path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
