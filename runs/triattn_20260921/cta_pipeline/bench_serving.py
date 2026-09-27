from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from opt_core.kernels import triattn_surround_tma as native
module=native._load();original=module.prologue
variants={'baseline':'off','pipeline':'on','selected':'auto'}
results={n:{'pro_rounds':[],'block_rounds':[]} for n in variants}
def pro():return pf.prologue(x,W,impl='fpf',ending=ending)
with torch.no_grad():
 os.environ['FPF_TRIATT_PRO_PIPELINE']='off'
 expected=call().clone();pieces=pro()
 for name,fn in variants.items():
  os.environ['FPF_TRIATT_PRO_PIPELINE']=fn;y=call().clone();actual=pro()
  torch.cuda.synchronize()
  assert torch.equal(y,expected),(name,'block',float((y.float()-expected.float()).abs().max()))
  assert all(torch.equal(t,r) for t,r in zip(actual,pieces)),(name,'prologue')
  kernels=cupti(call,5)
  assert any('triattn_m1_kernel' in k for k in kernels) if a.length>=512 else '_fwd' in kernels
  assert not any('flash_triattn' in k for k in kernels)
  assert any('pro_pipeline' in k for k in kernels) == (name=='pipeline' or (name=='selected' and a.length==384)), (name,kernels)
  results[name].update(bitwise_equal=True,kernels=kernels)
 checks=[]
 for seed,mode in [(901,'dense'),(902,'empty'),(903,'irregular')]:
  torch.manual_seed(seed);x.normal_()
  if mode=='dense':mask.fill_(True)
  elif mode=='empty':mask.fill_(False)
  else:mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.3)
  os.environ['FPF_TRIATT_PRO_PIPELINE']='off';reference=call().clone();reference_p=pro()
  os.environ['FPF_TRIATT_PRO_PIPELINE']='on';actual=call().clone();actual_p=pro()
  torch.cuda.synchronize()
  assert torch.equal(actual,reference),(mode,'block')
  assert all(torch.equal(t,r) for t,r in zip(actual_p,reference_p)),(mode,'prologue')
  checks.append(dict(seed=seed,mask=mode,bitwise_equal=True))
 # Restore the same distribution used in the original comparisons.
 torch.manual_seed(4103);x.normal_();mask.fill_(True);mask[:,::7]=False
 for rd in range(3):
  for name in (list(variants) if rd%2==0 else list(reversed(variants))):
   os.environ['FPF_TRIATT_PRO_PIPELINE']=variants[name]
   results[name]['pro_rounds'].append(graph_us(pro))
   results[name]['block_rounds'].append(graph_us(call))
   print('ROUND',rd,name,results[name]['pro_rounds'][-1],results[name]['block_rounds'][-1],flush=True)
 for r in results.values():
  r.update(pro_us=statistics.median(r['pro_rounds']),block_us=statistics.median(r['block_rounds']))
 os.environ.pop('FPF_TRIATT_PRO_PIPELINE',None)
out=dict(length=a.length,ending=ending,results=results,checks=checks)
Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
