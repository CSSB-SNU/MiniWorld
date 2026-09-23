"""Paired fixed-shape comparisons of baseline, LN/residual, dual GEMM and both."""
import argparse,json
from pathlib import Path
import torch
from miniworld_engine.autotune import cache
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bt
from miniworld_engine.modules import BidirectionalTriangleMultiplication
from check_trimul_f567_runtime import capture,paired

p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();L=a.length
report=dict(length=L,complete=False,cases={},cache=[],dynamic=False,
 device=torch.cuda.get_device_name(),torch_version=torch.__version__,
 correctness_dropout=0,timing_dropout=.25)
ln_fused,dual_fused=bt.input_ln_residual,bt.input_dual_bwd

def ln_split(x,w,b,eps):return bt.triton_layernorm(x,w,b,eps),x

def dual_split(g,f,w,v,length):
 out=torch.mm(g,w);out.addmm_(f,v);return out

subset=cache._cached_subset
original_miss=cache._miss
misses=[]

def reject_new_miss(op,*args,**kwargs):
 if op in ('trimul_input_ln_residual_bwd_triton','trimul_input_dual_bwd_triton'):
  misses.append((op,args[:3]))
  raise AssertionError(('new backward op cache miss',op,args[:3]))
 return original_miss(op,*args,**kwargs)

cache._miss=reject_new_miss

def observed(tuner,configs,nargs,meta):
 selected=subset(tuner,configs,nargs,meta)
 if tuner.fn.__name__ in ('_ln_bwd_residual_kernel','_input_dual_bwd_kernel'):
  assert selected,('new backward op cache miss',tuner.fn.__name__,cache.bucket_of_autotuner(tuner,nargs,meta))
  report['cache'].append(dict(kernel=tuner.fn.__name__,bucket=cache.bucket_of_autotuner(tuner,nargs,meta),candidates=[cache.as_cfg_dict(c) for c in selected]))
 return selected
cache._cached_subset=observed
refs=None;back_graphs={};full_graphs={};keep=[]
try:
 for label,ln,dual in [('baseline',False,False),('ln_residual',True,False),('dual',False,True),('both',True,True)]:
  bt.input_ln_residual=ln_fused if ln else ln_split
  bt.input_dual_bwd=dual_fused if dual else dual_split
  torch.compiler.reset();torch.manual_seed(911)
  model=BidirectionalTriangleMultiplication(128,implementation='triton',p_drop=0).cuda().bfloat16().train()
  with torch.no_grad():
   for name,w in model.named_parameters():
    if 'ln_' not in name:w.normal_(std=128**-.5)
  x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
  mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::3]=False;dy=torch.randn_like(x)
  fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  y=fn(x,mask);y.backward(dy)
  vals=[y.detach(),x.grad]+[w.grad for w in model.parameters()]
  assert all(v is not None and torch.isfinite(v).all() for v in vals)
  if refs is None:refs=[v.clone() for v in vals]
  errors=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(vals,refs)]
  assert max(errors)<.01,(label,errors)
  report['cases'][label]=dict(output_and_gradient_errors=errors,
   value_names=['output','input']+[n for n,_ in model.named_parameters()])
  del vals,y
  model.p_drop=.25
  # Autograd reuses the forward's stream. Build the retained forward on the
  # capture stream too; a default-stream forward cannot feed a side-stream
  # backward-only capture without an illegal legacy-stream dependency.
  backward_stream=torch.cuda.Stream();backward_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(backward_stream):saved_y=fn(x,mask)
  def backward_step(saved_y=saved_y,dy=dy,model=model,x=x):
   model.zero_grad(set_to_none=False);x.grad.zero_()
   saved_y.backward(dy,retain_graph=True)
   return x.grad
  with torch.cuda.stream(backward_stream):
   for _ in range(3):back_out=backward_step()
  torch.cuda.current_stream().wait_stream(backward_stream);torch.cuda.synchronize()
  backward_graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(backward_graph,stream=backward_stream):back_out=backward_step()
  back_graphs[label]=backward_graph
  def full_step(fn=fn,model=model,x=x,mask=mask,dy=dy):
   model.zero_grad(set_to_none=False);x.grad.zero_();y=fn(x,mask);y.backward(dy)
   return x.grad
  full_graphs[label],full_out=capture(full_step)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   backward_step();torch.cuda.synchronize()
  names=sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
  report['cases'][label]['cuda_kernels']=names
  report['cases'][label]['cuda_events']=[dict(name=e.name,duration_us=e.time_range.elapsed_us())
   for e in prof.events() if 'CUDA' in str(e.device_type)]
  if ln:assert any('_ln_bwd_residual_kernel' in n for n in names),names
  if dual:assert any('_input_dual_bwd_kernel' in n for n in names),names
  keep.append((model,x,mask,fn,dy,saved_y,back_out,full_out,backward_step,full_step))
  print('VERIFIED',label,'gradient',max(errors),flush=True)
 assert not misses,misses
 report['backward_with_grad_reset']=paired(back_graphs)
 report['forward_backward_with_grad_reset']=paired(full_graphs)
 report['complete']=True;a.output.write_text(json.dumps(report,indent=2));print('COMPLETE',a.length,report['backward_with_grad_reset'],report['forward_backward_with_grad_reset'],flush=True)
finally:
 bt.input_ln_residual=ln_fused;bt.input_dual_bwd=dual_fused;cache._cached_subset=subset;cache._miss=original_miss
