from pathlib import Path
import sys,os,json,ctypes
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint3 import Training
from wide_layout_ln import LayoutLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-layout-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;b=plan.b1;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();wanted=p.dt.clone()
 def old_ln():
  L=T._launch_module();drv=b.ln.unit.drv;args=L._Packed([b.lp])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(b.ln.handle)),b.grid,1,1,b.threads,1,1,b.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
 for mode in ('direct','direct_compact','transpose_compact'):
  for threads in (128,256):
   ln=LayoutLN(p,b,threads,mode)
   def old():plan.ln=None;return plan()
   def new():plan.ln=ln;return plan()
   p.dt.fill_(float('nan'));yn,gn=new();torch.cuda.synchronize()
   es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
   row=dict(mode=mode,threads=threads,cubin=str(ln.cubin),grid=ln.grid,errors=es,dt_error=error(p.dt,wanted))
   row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
   record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
   if row['strict']:
    row['times']=paired(dict(old_ln=old_ln,new_ln=ln,old_full=old,new_full=new))
    print('TIMES',mode,threads,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
   path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
