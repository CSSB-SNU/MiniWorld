from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from prefix_chunk_dx_ln import PrefixChunkDxLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
if D==256:from d256_gate_checkpoint import Training
else:from wide_checkpoint14 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-prefix-chunk-dx-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    old_reduce=plan.dx.reduce_only;oldrun=plan.schedule.run
    oldgemm=plan.dx.gemm_only if D==256 else plan.schedule.kernels['dx']
    for slots,minblocks in ([(3,2),(2,2)] if D==512 else [(3,3),(3,2)]):
        try:
            op=PrefixChunkDxLN(plan,slots,minblocks)
        except RuntimeError as exc:
            record['candidates'].append(dict(slots=slots,minblocks=minblocks,compile_error=str(exc)))
            print('COMPILE_ERROR',slots,minblocks,str(exc),flush=True)
            path.write_text(json.dumps(record,indent=2))
            continue
        item=dict(slots=slots,minblocks=minblocks,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,partial_bytes=op.partial.numel()*4)
        record['candidates'].append(item);print('LAUNCH',item,flush=True)
        def newrun(name):
            if name=='dx':op.gemm()
            else:oldrun(name)
        def install(new):
            plan.dx.reduce_only=op.finish if new else old_reduce
            if D==256:plan.dx.gemm_only=op.gemm if new else oldgemm
            else:plan.schedule.run=newrun if new else oldrun
        def old():install(False);return plan()
        def new():install(True);return plan()
        def oldstage():oldgemm();old_reduce()
        p.dx.fill_(float('nan'));yn,gn=new();torch.cuda.synchronize()
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['stage_times']=paired(dict(old_finish=oldstage,new_finish=op));gc.collect()
            item['full_times']=paired(dict(old_full=old,new_full=new));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in item[key].items()},flush=True)
        old()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
