from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_matched_partition_dw import MatchedPartitionDW
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=256;N=384;small_sms,splits,split_flags=((16,7,0),(20,7,1),(32,6,0))[index]
if D==256:from d256_permuted_ln_checkpoint import Training
else:from wide_checkpoint19 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,small_sms=small_sms,splits=splits,split_flags=split_flags)
path=THIS/f'result-d256-matched-partition-dw-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    op=MatchedPartitionDW(plan,small_sms,splits,split_flags);record['partition']=op.partition.verify_graph();print('PARTITION',record['partition']['sm_counts'],'source_grid',32*splits,'occupancy',op.occupancy,flush=True)
    def before():op.select(False);return plan()
    def after():op.select(True);return plan()
    yn,gn=after();torch.cuda.synchronize();actual=[x.clone() for x in [yn,*gn]]
    record['errors']={n:error(a,b) for n,a,b in zip(names,actual,expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',{k:v for k,v in record.items() if k!='partition'},flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        graph,out=capture(after);graph.replay();torch.cuda.synchronize()
        record['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],actual)}
        assert max(record['graph_errors'].values())<5e-6,record['graph_errors']
        del graph,out;gc.collect()
        def oldbw():op.select(False);return plan.backward()
        def newbw():op.select(True);return plan.backward()
        record['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
        record['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
        print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
