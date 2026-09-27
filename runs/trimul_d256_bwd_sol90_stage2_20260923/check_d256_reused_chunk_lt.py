"""Bound GP chunks near L2 capacity, counting all partial-reduction costs."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_spatial_checkpoint import Training
from d256_reused_chunk_lt import ReusedChunkLt
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));splits,chunks=((64,8),(128,16),(96,24))[index//2];evict_last=bool(index%2)
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
record=dict(D=256,L=384,splits=splits,chunks=chunks,evict_last=evict_last,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-reused-chunk-lt-L384-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();dxn=p.tensors[10].clone()
    source=plan.late_output_dw.oldsource;matrix=plan.dx.matrix_product;reduce=plan.dx.reduce_only
    op=ReusedChunkLt(plan,splits,chunks,evict_last)
    op.input.copy_(plan.dx.input[:,:op.rows]);probe=op.dx[0];candidates={}
    record['algorithm_errors']={}
    for candidate in probe.indices:
        probe.index=candidate;probe();torch.cuda.synchronize()
        err=error(p.tensors[10][:op.rows],dxn[:op.rows]);record['algorithm_errors'][candidate]=err
        if err==0:
            def run(candidate=candidate):probe.index=candidate;probe()
            candidates[f'algorithm_{candidate}']=run
    assert candidates,'No exact compact dX heuristic'
    times=paired(candidates);chosen=min(times,key=lambda key:times[key]['median_us']);probe.index=int(chosen.split('_')[-1])
    record['algorithm_times']={k:v['median_us'] for k,v in times.items()};record['algorithm']=list(probe.heuristics[probe.index].algo.data)
    op.set_algorithm(record['algorithm'])
    record.update(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,gp_chunk_bytes=op.input.numel()*2,partial_bytes=op.partial.numel()*4,ranges=op.ranges)
    def select(new):
        plan.late_output_dw.oldsource=op if new else source
        plan.dx.matrix_product=(lambda:None) if new else matrix
        plan.dx.reduce_only=op.reduce if new else reduce
    def before():select(False);return plan()
    def after():select(True);return plan()
    p.gp_all.fill_(float('nan'));p.tensors[10].fill_(float('nan'));op.partial.fill_(float('nan'))
    yn,gn=after();torch.cuda.synchronize()
    record['gp_error']=error(torch.stack(op.gp),gp[...,op.ranges[-1][0]:op.ranges[-1][1]]);record['dxn_error']=error(p.tensors[10],dxn)
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        eager=[x.clone() for x in [yn,*gn]]
        capture=torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):after()
        p.gp_all.fill_(float('nan'));p.tensors[10].fill_(float('nan'));op.partial.fill_(float('nan'))
        capture.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[plan.f.output.y,*p.outputs],eager)}
        assert max(record['graph_errors'].values())<5e-6,record['graph_errors']
        del capture,eager;gc.collect()
        def oldbw():select(False);return plan.backward()
        def newbw():select(True);return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
