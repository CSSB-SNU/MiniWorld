"""Full native gradient and captured full-workload test of large dX chunks."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_chunked_lt_overlap import ChunkedLtOverlap
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-chunked-lt-overlap-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone();dxn=p.tensors[10].clone()
    source=plan.b7.source_only;matrix=plan.dx.matrix_product;reduce=plan.dx.reduce_only
    for splits,chunks in ((32,4),(32,2),(16,2),(8,2)):
        op=ChunkedLtOverlap(plan,splits,chunks)
        for overlap in (True,False):
            op.overlap=overlap
            item=dict(splits=splits,chunks=chunks,overlap=overlap,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes)
            record['candidates'].append(item)
            def select(new):
                plan.b7.source_only=op if new else source
                plan.dx.matrix_product=(lambda:None) if new else matrix
                plan.dx.reduce_only=op.reduce if new else reduce
            def before():select(False);return plan()
            def after():select(True);return plan()
            p.gp_all.fill_(float('nan'));p.tensors[10].fill_(float('nan'))
            try:
                yn,gn=after();torch.cuda.synchronize()
                item['gp_error']=error(p.gp_all,gp);item['dxn_error']=error(p.tensors[10],dxn)
                item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
                item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
                print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
                if item['strict'] and not op.local_bytes:
                    eager=[x.clone() for x in [yn,*gn]]
                    capture=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(capture):after()
                    capture.replay();torch.cuda.synchronize()
                    item['graph_errors']={n:error(a,b) for n,a,b in zip(names,[plan.f.output.y,*p.outputs],eager)}
                    assert max(item['graph_errors'].values())<5e-6,item['graph_errors']
                    del capture,eager;gc.collect()
                    def oldbw():select(False);return plan.backward()
                    def newbw():select(True);return plan.backward()
                    item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
                    item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
                    print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in item[key].items()},flush=True)
            except RuntimeError as exc:
                item['failure']=str(exc);print('FAIL',item,flush=True)
            before();path.write_text(json.dumps(record,indent=2))
        for gemm in op.dx:gemm.close()
        del op;gc.collect()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
