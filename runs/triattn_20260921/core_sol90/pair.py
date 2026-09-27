"""Paired replay timing; alternate ordering to reduce drift between variants."""
from pathlib import Path
import os, statistics, json
HERE=Path(__file__).resolve().parent
exec(compile((HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0],str(HERE/'bench.py'),'exec'))
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 graphs={}
 outputs={}
 for scope,fn in [('core',core_call),('block',call)]:
  graphs[scope]={};outputs[scope]={}
  for name,core in variants.items():
   KW['core']=core
   for _ in range(5):fn()
   torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn()
   graphs[scope][name]=graph
   graph.replay();outputs[scope][name]=out.clone()
  torch.cuda.synchronize()
  assert torch.equal(outputs[scope]['serving'],outputs[scope][mode]),scope
 data={scope:{name:[] for name in variants} for scope in graphs}
 events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(2)]
 for scope,gg in graphs.items():
  for _ in range(100):
   for graph in gg.values():graph.replay()
  torch.cuda.synchronize()
  for rd in range(16):
   order=list(variants) if rd%2==0 else list(reversed(variants))
   for ei,name in enumerate(order):
    st,en=events[ei]
    for _ in range(3):gg[name].replay()
    st.record()
    for _ in range(20):gg[name].replay()
    en.record()
   torch.cuda.synchronize()
   for ei,name in enumerate(order):data[scope][name].append(events[ei][0].elapsed_time(events[ei][1])*1000/20)
  print(scope,json.dumps(data[scope]),flush=True)
 summary={scope:{'serving_us':statistics.median(d['serving']),'candidate_us':statistics.median(d[mode]),'paired_ratio':statistics.median([c/s for c,s in zip(d[mode],d['serving'])])} for scope,d in data.items()}
 Path(a.output).write_text(json.dumps({'mode':mode,'length':a.length,'ending':ending,'summary':summary,'rounds':data},indent=2))
 print('RESULT',json.dumps(summary),flush=True)
