"""Current qualified short paths: profiler trace or separate graph diagnostics."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=(256,384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=384
if D==256:from d256_stats_checkpoint import Training
else:from wide_checkpoint18 import Training
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy)
    for _ in range(3):plan()
    torch.cuda.synchronize()
    if os.environ.get('MEASURE_COMPONENTS')!='1':
        torch.cuda.cudart().cudaProfilerStart();plan.backward();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
    else:
        s=plan.schedule
        if D==256:
            ops=dict(output_epi=plan.b1.prepare,dnorm=lambda:s.run('dn'),output_ln=plan.b1.ln,
                     dw_proj=lambda:s.run('dwp'),dw_gate=lambda:s.run('dwg'),
                     **{name:(lambda name=name:s.run(name)) for name in ('bc0','bc1','bc2','bc3')},
                     source=plan.b7.source_only,dx_gemm=plan.dx.gemm_only,input_finish=plan.dx.reduce_only)
        else:
            ops=dict(output_epi=plan.prefix_gate,dnorm=lambda:s.run('dn'),output_ln=plan.ln,
                     dw_proj=lambda:s.run('dwp'),contract_gp=plan.contract_gp,input_dw=plan.input_dw,
                     dx_gemm=lambda:s.run('dx'),input_finish=plan.dx.reduce_only)
            if plan.delta_output is not None:ops.update(delta_norm=plan.delta_backward.launch,delta_projection=plan.delta_projection)
        record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,diagnostic_only=True)
        record['components']=paired(ops);gc.collect()
        record['workload']=paired(dict(backward=plan.backward,full=plan));gc.collect()
        record['complete']=True
        (THIS/f'components-current-short-D{D}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
        print('COMPONENTS',{k:round(v['median_us'],3) for k,v in record['components'].items()},flush=True)
        print('WORKLOAD',{k:round(v['median_us'],3) for k,v in record['workload'].items()},flush=True)
