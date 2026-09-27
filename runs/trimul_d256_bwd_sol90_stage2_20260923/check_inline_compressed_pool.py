"""Compare identical kernels/bits using opt-in hardware-compressible storage."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from inline_compressed_pool import InlineCompressedPool
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=(256,384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=384
if D==256:from d256_permuted_ln_checkpoint import Training
else:from wide_checkpoint19 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-inline-compressed-pool-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    old=Training(leaves,mask,ds,dy);y,g=old();expected=[x.clone() for x in [y,*g]]
    pool=InlineCompressedPool()
    with pool.context():new=Training(leaves,mask,ds,dy)
    record['allocation_stats']=pool.stats()
    assert record['allocation_stats']['allocations']>0
    assert record['allocation_stats']['allocations']==record['allocation_stats']['compressible_allocations']
    assert record['allocation_stats']['errors']==0
    yn,gn=new();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        eager=[x.clone() for x in [yn,*gn]]
        capture=torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):yn,gn=new()
        capture.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],eager)}
        assert max(record['graph_errors'].values())<5e-6,record['graph_errors']
        del capture,eager;gc.collect()
        record['backward_times']=paired(dict(old_backward=old.backward,new_backward=new.backward));gc.collect()
        record['full_times']=paired(dict(old_full=old,new_full=new));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in record[key].items()},flush=True)
        record['component_times']=paired(dict(old_source=old.b7.source_only,new_source=new.b7.source_only,
            old_ln=old.b1.ln if D==256 else old.ln,new_ln=new.b1.ln if D==256 else new.ln))
        print('COMPONENTS',{k:v['median_us'] for k,v in record['component_times'].items()},flush=True)
    record['allocation_stats_after']=pool.stats()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
