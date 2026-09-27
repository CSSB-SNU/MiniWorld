"""Three-way paired replay with short batches and balanced permutations.

Same data and exact full-output gates as pair.py. This reduces the interval
between candidate/reference samples after initial long-batch clock variation.
"""
from pathlib import Path
import importlib.util,itertools,json,os,statistics
import numpy as np
HERE=Path(__file__).resolve().parent
os.environ['SOL_VARIANT']='baseqthreehalf0'
exec(compile((HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0],str(HERE/'bench.py'),'exec'))

other='baseqthreehalf1';name='triattn_sol_'+other
spec=importlib.util.spec_from_file_location(name,HERE/'build'/name/(name+'.so'))
ext1=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext1)
spec=importlib.util.spec_from_file_location('q_three_wrapper1',HERE/other/'triattn_m1.py')
wrap1=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrap1);wrap1._EXT=ext1
def candidate1(q,k,v,b,mask,scale):
 return wrap1.triangle_attention_m1(q,k,v,b.reshape(1,1,4,a.length,a.length),mask=mask,scale=scale).reshape_as(q)
variants[other]=candidate1
permutations=list(itertools.permutations(variants))
summary={};all_rounds={}
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 for scope,fn in [('core',core_call),('block',call)]:
  graphs={};outputs={}
  for n,core in variants.items():
   KW['core']=core
   for _ in range(5):fn()
   torch.cuda.synchronize()
   gr=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gr):out=fn()
   graphs[n]=gr
   gr.replay();outputs[n]=out.clone()
  torch.cuda.synchronize()
  assert all(torch.equal(outputs['serving'],outputs[n]) for n in variants),(scope,'full output mismatch')
  for _ in range(600):
   for gr in graphs.values():gr.replay()
  torch.cuda.synchronize()
  records={n:[] for n in variants}
  # Each round has8 subrounds,2 replays/measurement. Ordering cycles through
  # all6 permutations;72 rounds make each subround position exactly balanced.
  events=[[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in variants] for _ in range(8)]
  for rd in range(72):
   orders=[]
   for sub in range(8):
    order=permutations[(rd+sub)%6];orders.append(order)
    for ei,n in enumerate(order):
     st,en=events[sub][ei]
     st.record()
     graphs[n].replay();graphs[n].replay()
     en.record()
   torch.cuda.synchronize()
   measured={n:[] for n in variants}
   for sub,order in enumerate(orders):
    for ei,n in enumerate(order):
     st,en=events[sub][ei];measured[n].append(st.elapsed_time(en)*500.)
   for n in variants:records[n].append(statistics.mean(measured[n]))
  rng=np.random.default_rng(260922)
  comparisons={}
  for n in variants:
   if n=='serving':continue
   ratios=np.array(records[n])/np.array(records['serving'])
   boot=np.median(ratios[rng.integers(len(ratios),size=(20000,len(ratios)))],axis=1)
   comparisons[n]=dict(serving_us=statistics.median(records['serving']),candidate_us=statistics.median(records[n]),
      paired_ratio=float(np.median(ratios)),paired_median_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
  summary[scope]=comparisons;all_rounds[scope]=records
  print('INTERLEAVED',scope,json.dumps(comparisons),flush=True)
Path(a.output).write_text(json.dumps(dict(ending=ending,length=a.length,full_bitwise=True,
 rounds=72,subrounds=8,replays_per_sample=2,summary=summary,timings=all_rounds),indent=2)+'\n')
