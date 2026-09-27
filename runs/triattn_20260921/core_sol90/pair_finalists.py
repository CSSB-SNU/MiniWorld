"""Same-process comparison of installed, fixed descriptors, and partial-Q fusion."""
from pathlib import Path
import os,json,statistics,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
exec(compile((HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0],str(HERE/'bench.py'),'exec'))
peer='basef40fixeddesc';name='triattn_sol_'+peer
spec=importlib.util.spec_from_file_location(name,HERE/'build'/name/(name+'.so'))
peerext=importlib.util.module_from_spec(spec);spec.loader.exec_module(peerext)
spec=importlib.util.spec_from_file_location('fixed_descriptor_wrapper',HERE/peer/'triattn_m1.py')
peermod=importlib.util.module_from_spec(spec);spec.loader.exec_module(peermod);peermod._EXT=peerext
def peer_call(q,k,v,b,mask,scale):
 return peermod.triangle_attention_m1(q,k,v,b.reshape(1,1,4,a.length,a.length),mask=mask,scale=scale).reshape_as(q)
variants[peer]=peer_call
baseline_sha=hashlib.sha256((HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so').read_bytes()).hexdigest()
assert baseline_sha=='6fe43326e87c87bb5512b39b66437493123e6e68cdc6799bdcc7659f6173f694'
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 graphs={};data={}
 for scope,fn in [('core',core_call),('block',call)]:
  graphs[scope]={};outs={};data[scope]={n:[] for n in variants}
  for n,fncore in variants.items():
   KW['core']=fncore
   for _ in range(5):fn()
   torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):out=fn()
   graphs[scope][n]=cg;outs[n]=out;cg.replay()
  torch.cuda.synchronize()
  assert all(torch.equal(outs['serving'],out) for out in outs.values()),scope
 events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in variants]
 for scope,gg in graphs.items():
  for _ in range(100):
   for cg in gg.values():cg.replay()
  torch.cuda.synchronize()
  names=list(variants)
  for rd in range(18):
   order=names[rd%3:]+names[:rd%3]
   if rd%2:order=list(reversed(order))
   for i,n in enumerate(order):
    for _ in range(3):gg[n].replay()
    start,end=events[i];start.record()
    for _ in range(20):gg[n].replay()
    end.record()
   torch.cuda.synchronize()
   for i,n in enumerate(order):data[scope][n].append(events[i][0].elapsed_time(events[i][1])*1000/20)
 summary={}
 for scope,d in data.items():
  summary[scope]={'median_us':{n:statistics.median(xx) for n,xx in d.items()},
   'composite_over_fixed':statistics.median([v/f for v,f in zip(d[mode],d[peer])]),
   'fixed_over_serving':statistics.median([v/f for v,f in zip(d[peer],d['serving'])]),
   'composite_over_serving':statistics.median([v/f for v,f in zip(d[mode],d['serving'])])}
 result={'baseline_sha':baseline_sha,'mode':mode,'peer':peer,'ending':ending,'length':a.length,'summary':summary,'rounds':data}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(summary),flush=True)
