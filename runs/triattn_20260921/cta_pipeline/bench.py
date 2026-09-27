from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from build import build
from opt_core.kernels import triattn_surround_tma as native
ext=build();module=native._load();original=module.prologue
variants={'baseline':original,'pipeline64':ext.pro64,'pipeline128':ext.pro128}
results={n:{'pro_rounds':[],'block_rounds':[]} for n in variants}
def pro():return pf.prologue(x,W,impl='fpf',ending=ending)
with torch.no_grad():
 expected=call().clone();pieces=pro()
 for name,fn in variants.items():
  module.prologue=fn;y=call().clone();actual=pro()
  torch.cuda.synchronize()
  assert torch.equal(y,expected),(name,'block',float((y.float()-expected.float()).abs().max()))
  assert all(torch.equal(t,r) for t,r in zip(actual,pieces)),(name,'prologue')
  kernels=cupti(call,5)
  assert any('triattn_m1_kernel' in k for k in kernels) if a.length>=512 else '_fwd' in kernels
  assert not any('flash_triattn' in k for k in kernels)
  results[name].update(bitwise_equal=True,kernels=kernels)
 for rd in range(3):
  for name in (list(variants) if rd%2==0 else list(reversed(variants))):
   module.prologue=variants[name]
   results[name]['pro_rounds'].append(graph_us(pro))
   results[name]['block_rounds'].append(graph_us(call))
   print('ROUND',rd,name,results[name]['pro_rounds'][-1],results[name]['block_rounds'][-1],flush=True)
 for r in results.values():
  r.update(pro_us=statistics.median(r['pro_rounds']),block_us=statistics.median(r['block_rounds']))
 module.prologue=original
out=dict(length=a.length,ending=ending,results=results)
Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
