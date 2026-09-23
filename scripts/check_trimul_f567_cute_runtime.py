"""Check the actual CuTe bidirectional wiring, saved gradients and forward graphs."""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
from miniworld_engine.autotune import native
from miniworld_engine.kernels.trimul_inproj.cute import bidir_training as bc
from miniworld_engine.kernels.trimul_inproj.cute.output_f567 import output_f567_sm90
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize,_te_forward
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_train
from miniworld_engine.modules import BidirectionalTriangleMultiplication
from check_trimul_f567_runtime import capture,paired

p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();L=a.length
report=dict(length=L,complete=False,cache=[])
original_select=native.select_config

def select(op,**kwargs):
 result=original_select(op,**kwargs)
 if op=='trimul_output_f567_sm90_cute':
  assert result,'F567 CuTe runtime cache miss'
  report['cache'].append(dict(bucket=kwargs['bucket'],selected=result))
 return result
native.select_config=select
name='miniworld_engine.kernels.trimul_inproj.cute.bidir_f567_baseline'
spec=importlib.util.spec_from_file_location(name,a.baseline);baseline=importlib.util.module_from_spec(spec);sys.modules[name]=baseline;spec.loader.exec_module(baseline)
fused=bc.BidirBackHalf;refs=None;graphs={};keep=[]
try:
 for label,cls in [('h100_split',baseline.BidirBackHalf),('h100_f567',fused)]:
  bc.BidirBackHalf=cls;torch.compiler.reset();torch.manual_seed(733)
  model=BidirectionalTriangleMultiplication(128,implementation='cute',p_drop=0).cuda().bfloat16().train()
  with torch.no_grad():
   for name,w in model.named_parameters():
    if 'ln_' not in name:w.normal_(std=128**-.5)
    elif 'bias' in name:w.normal_(std=.1)
  x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
  mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::3]=False
  dy=torch.randn_like(x)
  fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  y=fn(x,mask);y.backward(dy)
  values=[y.detach(),x.grad]+[w.grad for w in model.parameters()]
  assert all(v is not None and torch.isfinite(v).all() for v in values)
  if refs is None:refs=[v.clone() for v in values]
  errors=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(values,refs)]
  assert max(errors)<.015,errors
  report.setdefault('output_and_gradient_errors',{})[label]=errors
  if label=='h100_f567':
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    fn(x,mask)
    torch.cuda.synchronize()
   report['cuda_kernels']=sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
   assert any('F567Sm90' in name for name in report['cuda_kernels']),report['cuda_kernels']

  model.p_drop=.25;graph,out=capture(lambda:fn(x,mask));graph.replay();torch.cuda.synchronize()
  old=out.clone();graph.replay();torch.cuda.synchronize();assert not torch.equal(old,out),'dropout froze'
  graphs[label]=graph;keep.append((model,x,mask,fn,out))
 report['whole_forward']=paired(graphs)
finally:bc.BidirBackHalf=fused
assert report['cache'],'compiled model never reached F567 CuTe'
del graphs,keep,refs;torch.cuda.empty_cache()
kw=dict(device='cuda',dtype=torch.bfloat16);m=L*L;torch.manual_seed(312)
view=torch.randn(256,m,**kw).t();x=torch.randn(m,128,**kw)
wp=torch.randn(128,256,**kw)/16;wg=torch.randn(128,128,**kw)/128**.5
res=torch.randn(m,128,**kw);ds=(torch.rand(L,128,device='cuda')>.25).to(torch.bfloat16)/.75
g=torch.ones(256,device='cuda');b=torch.zeros_like(g)

def tail(kind):
 if kind=='split':
  proj,norm,mean,rstd=_te_forward(view,g,b,wp,None,1e-5)
  y,gate=gate_elem_train(x,proj,wg,res,ds,L)
 else:
  norm,mean,rstd=_ln_materialize(view,g,b,1e-5)
  y,proj,gate=(output_f567_train if kind=='triton' else output_f567_sm90)(norm,x,wp,wg,res,ds,L)
 return y,proj,gate,norm,mean,rstd

graphs={};keep=[];refs=None
for kind in ['split','triton','cute']:
 graph,out=capture(lambda:tail(kind));graph.replay();torch.cuda.synchronize()
 if refs is None:refs=[v.clone() for v in out[:3]]
 errors=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(out[:3],refs)]
 assert max(errors)<.004,errors
 report.setdefault('tail_errors',{})[kind]=errors;graphs[kind]=graph;keep.append(out)
report['tail_forward']=paired(graphs);report['complete']=True
a.output.write_text(json.dumps(report,indent=2));print('COMPLETE',report['tail_forward'],flush=True)
