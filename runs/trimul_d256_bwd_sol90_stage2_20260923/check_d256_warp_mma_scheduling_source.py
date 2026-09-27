"""Check exact compressed sigmoid correction and complete workload timing."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from d256_warp_mma_pipe_source import WarpMMAPipeSource
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));consumer,opt,static_slots=((88,2,True),(88,1,True),(88,3,False),(96,2,False))[index]
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
record=dict(D=256,L=384,consumer=consumer,opt=opt,static_slots=static_slots,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-warp-mma-scheduling-source-L384-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    part=p.floats[7].reshape(-1)[3*256*256:].as_strided((plan.b7.splits,8*256*256),(11*256*256,1));partial=part.clone()
    source=plan.late_output_dw.oldsource
    op=WarpMMAPipeSource(plan,consumer,opt,static_slots)
    record['pool_metadata']=op.pool_metadata
    print('RESOURCE',dict(registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy),flush=True)
    record.update(cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy)
    def before():plan.late_output_dw.oldsource=source;return plan()
    def after():plan.late_output_dw.oldsource=op;return plan()
    p.gp_all.fill_(float('nan'));part.fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
    record['gp_error']=error(p.gp_all,gp);record['partial_error']=error(part,partial)
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        candidate_eager=[x.clone() for x in [yn,*gn]]
        graph,out=capture(after);p.gp_all.fill_(float('nan'));part.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],candidate_eager)}
        assert all(v<(1e-30 if n=='y' else 5e-6) for n,v in record['graph_errors'].items())
        del graph,out;gc.collect()
        record['stage_times']=paired(dict(old_source=source,new_source=op));gc.collect()
        def oldbw():plan.late_output_dw.oldsource=source;return plan.backward()
        def newbw():plan.late_output_dw.oldsource=op;return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
