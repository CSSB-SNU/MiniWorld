from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint23 import Training
from wide_spatial_gp_dx_overlap import SpatialGPDX
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//3];variant=index%3
leaves,dy,mask,ds,ref,triton,names=setup(D,384)
record=dict(D=D,L=384,overlap=variant>0,dw_overlap=variant==2,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-spatial-gp-dx-overlap-D{D}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();dxn=p.tensors[10].clone()
    oldsource=plan.b7.source_only;oldrun=plan.schedule.run;oldpack=plan.dx.pack_weights
    op=SpatialGPDX(plan,variant>0,variant==2)
    def run(name):
        if name!='dx':oldrun(name)
    def restore():plan.b7.source_only=oldsource;plan.schedule.run=oldrun;plan.dx.pack_weights=oldpack
    def attach():plan.b7.source_only=op;plan.schedule.run=run;plan.dx.pack_weights=lambda:None
    def before():restore();return plan()
    def after():attach();return plan()
    p.gp_all.fill_(float('nan'));p.tensors[10].fill_(float('nan'));yn,gn=after();torch.cuda.synchronize()
    record.update(registers=op.registers,local_bytes=op.local_bytes,cubin=str(op.cubin),gp_error=error(p.gp_all,gp),dxn_error=error(p.tensors[10],dxn))
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict'] and not op.local_bytes:
        graph,out=capture(after);p.gp_all.fill_(float('nan'));p.tensors[10].fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],expected)}
        assert all(v<(1e-30 if n=='y' else 5e-6) for n,v in record['graph_errors'].items()),record['graph_errors']
        del graph,out;gc.collect()
        def oldbw():restore();return plan.backward()
        def newbw():attach();return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for field in ['backward_times','full_times'] for k,v in record[field].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
