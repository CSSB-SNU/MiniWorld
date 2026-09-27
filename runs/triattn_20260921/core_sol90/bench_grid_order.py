from pathlib import Path
import os,json,statistics,hashlib
HERE=Path(__file__).resolve().parent
exec(compile((HERE/'screen_grid_order.py').read_text(),str(HERE/'screen_grid_order.py'),'exec'))
os.environ['SOL_VARIANT']='basegrid'
s=(HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(s,str(HERE/'bench.py'),'exec'))
def ordered_candidate(order):
 def f(*args):
  os.environ['TRIATTN_SOL_GRID_ORDER']=str(order)
  return candidate(*args)
 return f
variants={'serving':'tier:triattn_native'}
orders=[int(i) for i in os.environ.get('GRID_ORDERS','0,1,2,3,4,5,6,7,8,9').split(',')]
variants.update({'grid%d'%i:ordered_candidate(i) for i in orders})
baseline_sha=hashlib.sha256((HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so').read_bytes()).hexdigest()
assert baseline_sha=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 graphs={};data={};kernels={}
 for scope,fn in [('core',core_call),('block',call)]:
  graphs[scope]={};outs={};data[scope]={n:[] for n in variants}
  for n,fncore in variants.items():
   KW['core']=fncore
   for _ in range(5):fn()
   torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):out=fn()
   graphs[scope][n]=cg;outs[n]=out;cg.replay()
   if scope=='block':
    kk=cupti(fn,5)
    expected='ta_core_broadcast::triattn_m1_kernel' if n=='serving' else 'ta_sol_basegrid::triattn_m1_kernel'
    assert any(expected in key for key in kk) and not any('flash_triattn' in key for key in kk),kk
    kernels[n]=kk
  torch.cuda.synchronize()
  assert all(torch.equal(outs['serving'],out) for out in outs.values()),scope
  print('BITWISE ALL',scope,flush=True)
 names=list(variants)
 events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in names]
 for scope,gg in graphs.items():
  for _ in range(100):
   for cg in gg.values():cg.replay()
  torch.cuda.synchronize()
  for rd in range(max(22,4*len(names))):
   order=names[rd%len(names):]+names[:rd%len(names)]
   if rd%2:order=list(reversed(order))
   for i,n in enumerate(order):
    for _ in range(3):gg[n].replay()
    start,end=events[i];start.record()
    for _ in range(20):gg[n].replay()
    end.record()
   torch.cuda.synchronize()
   for i,n in enumerate(order):data[scope][n].append(events[i][0].elapsed_time(events[i][1])*1000/20)
  print('TIMED',scope,flush=True)
 summary={}
 for scope,d in data.items():
  summary[scope]={'median_us':{n:statistics.median(xx) for n,xx in d.items()},
   'over_serving':{n:statistics.median([v/f for v,f in zip(xx,d['serving'])]) for n,xx in d.items()},
   'over_grid0':{n:statistics.median([v/f for v,f in zip(xx,d['grid0'])]) for n,xx in d.items()}}
 result={'baseline_sha':baseline_sha,'ending':ending,'length':a.length,'core_block_bitwise':True,'summary':summary,'rounds':data,'kernels':kernels}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(summary),flush=True)
