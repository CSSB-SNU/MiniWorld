"""Screen only exact-GP projection algorithms, then time the complete source."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from d256_dense_recompute_source import DenseRecomputeSource
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
record=dict(D=256,L=384,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-dense-recompute-source-L384-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    source=plan.late_output_dw.oldsource;op=DenseRecomputeSource(plan);part=op.partial.clone()
    def option(mat,index,epi=False):
        def run():
            mat.index=index;mat()
            if epi:op.epi()
        return run
    valid={};record['projection_screen']={}
    for index in op.products.indices:
        op.products.index=index;op.products();op.epi();v=error(p.gp_all,gp)
        record['projection_screen'][index]=v
        if v==0:valid[str(index)]=option(op.products,index)
    print('PROJECTION_SCREEN',record['projection_screen'],flush=True);path.write_text(json.dumps(record,indent=2))
    if not valid:raise RuntimeError('No projection algorithm retains exact GP')
    times=paired(valid);record['projection_times']=times;gc.collect()
    op.products.index=int(min(times,key=lambda k:times[k]['median_us']));op.products();op.epi()
    valid={};record['weight_screen']={}
    for index in op.weights.indices:
        op.weights.index=index;op.weights();v=error(op.partial,part)
        record['weight_screen'][index]=v
        if v<2e-5:valid[str(index)]=option(op.weights,index)
    print('WEIGHT_SCREEN',record['weight_screen'],flush=True);path.write_text(json.dumps(record,indent=2))
    assert valid
    times=paired(valid);record['weight_times']=times;gc.collect()
    op.weights.index=int(min(times,key=lambda k:times[k]['median_us']))
    record['selected']=dict(projection=op.products.index,weights=op.weights.index)
    def before():plan.late_output_dw.oldsource=source;return plan()
    def after():plan.late_output_dw.oldsource=op;return plan()
    yn,gn=after();torch.cuda.synchronize();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',{k:record[k] for k in ['selected','strict','errors']},flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        candidate_eager=[x.clone() for x in [yn,*gn]]
        graph,out=capture(after);p.gp_all.fill_(float('nan'));op.pre.fill_(float('nan'));op.partial.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],candidate_eager)}
        assert all(v<(1e-30 if n=='y' else 5e-6) for n,v in record['graph_errors'].items())
        del graph,out;gc.collect()
        def oldbw():plan.late_output_dw.oldsource=source;return plan.backward()
        def newbw():plan.late_output_dw.oldsource=op;return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['times']=paired(dict(old_source=source,new_source=op,old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
