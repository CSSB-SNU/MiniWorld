"""Build each backward fusion using the standard Triton CSV/autotune cache."""
import argparse,json,math,os
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--op',choices=['ln','dual'],required=True);p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.environ.update(MINIWORLD_DRIVER_LENGTH=str(a.length),MINIWORLD_DRIVER_WIDTH='128',MINIWORLD_DRIVER_HEADS='128')
 import torch
 from miniworld_engine import settings
 from miniworld_engine.autotune import capture,cache
 from miniworld_engine.kernels.trimul_inproj.triton import backward_fused as f
 from miniworld_engine.kernels.drivers.trimul_backward import ln_residual,dual
 settings.configure(run_autotune=True,capture=True,compile_jobs=8,bench_clear_mb=16,bench_rep_ms=8)
 capture.install();capture.set_incremental(False)
 kernel=f._ln_bwd_residual_kernel if a.op=='ln' else f._input_dual_bwd_kernel
 original=kernel._bench;refs=None;checked=[]

 def bench(*args,config,**meta):
  nonlocal refs
  # Capture installs its event timer on first run. Override that timer here,
  # after its setup, while retaining standard config recording/publication.
  kernel.__dict__['do_bench']=lambda call,quantiles,**kw: __import__('triton').testing.do_bench_cudagraph(call,rep=8,quantiles=quantiles)
  timings=original(*args,config=config,**meta);row=dict(config=cache.as_cfg_dict(config),timings=timings)
  if math.isfinite(timings[0]):
   v=kernel.nargs
   if refs is None:
    if a.op=='ln':
     x,dy,w,mean,rs,dr=(v[n].float() for n in ['X','DY','W','Mean','Rstd','RES'])
     xh=(x-mean[:,None])*rs[:,None];u=dy*w
     dx=rs[:,None]*(u-u.mean(1,keepdim=True)-xh*(u*xh).mean(1,keepdim=True))
     refs=(dx.to(v['X'].dtype).float()+dr,(dy*xh).sum(0),dy.sum(0))
    else:
     g,ff,w,ww=(v[n].float() for n in ['G','F','W','V'])
     refs=((g@w).to(v['G'].dtype).float()+ff@ww,)
   values=[v[n] for n in (['DX','DW','DB'] if a.op=='ln' else ['Y'])]
   errors=[float((val.float()-ref).norm()/ref.norm().clamp_min(1e-8)) for val,ref in zip(values,refs)]
   assert all(torch.isfinite(val).all() for val in values) and max(errors)<.004,(config,errors)
   row['relative_l2']=errors
  checked.append(row)
  if len(checked)%64==0:print('CHECKED',len(checked),flush=True)
  return timings

 kernel._bench=bench
 a.output.parent.mkdir(exist_ok=True,parents=True)
 try:
  (ln_residual if a.op=='ln' else dual)()
  assert checked and not capture.record_errors(),capture.record_errors()
  capture.dump_shard(str(a.output.with_suffix('.shard.json')),unit_complete=True)
  written=capture.flush()
  result=dict(method='cuda_graph',complete=True,op=a.op,length=a.length,grid_count=len(kernel.configs),measured_count=len(checked),configs=checked,winner=cache.as_cfg_dict(kernel.best_config),written=written)
  a.output.write_text(json.dumps(result,indent=2,default=str));print('COMPLETE',a.op,a.length,result['winner'],flush=True)
 finally:capture.shutdown_precompile()


if __name__=='__main__':main()
