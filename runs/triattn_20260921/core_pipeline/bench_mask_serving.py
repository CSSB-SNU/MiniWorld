from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from opt_core.kernels import triattn_surround_tma as bridge
from types import SimpleNamespace
ext=SimpleNamespace(stage=bridge._load().stage_broadcast_mask)
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core():return pf.core_attention(q,k,v,b,m5,core='tier:triattn_native')
 expected=call().clone();expected_c=core().clone()
 native=sys.modules['triattn_m1']._EXT;original=native.stage_mask
 variants={'baseline':original,'broadcast':ext.stage};results={n:dict(core_rounds=[],block_rounds=[],stage_rounds=[]) for n in variants}
 counts=torch.zeros(2,device='cuda',dtype=torch.int32)
 for mode in ['dense','empty','irregular','prefix']:
  if mode=='dense':mask.fill_(True)
  elif mode=='empty':mask.fill_(False)
  elif mode=='irregular':mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.3)
  else:mask.fill_(True);mask[:,:73]=False;mask[:,a.length-27:]=False
  c0=torch.zeros_like(counts);c1=torch.zeros_like(counts)
  t0=original(m5,c0);t1=ext.stage(m5,c1)
  assert all(torch.equal(x,y) for x,y in zip(t0,t1)),mode
  assert torch.equal(c0,c1),('counts',mode)
  os.environ['FPF_TRIATT_MASK_STAGE']='off';y0=call().clone();o0=core().clone()
  os.environ['FPF_TRIATT_MASK_STAGE']='on';y1=call().clone();o1=core().clone()
  assert torch.equal(y0,y1) and torch.equal(o0,o1),('block/core',mode)
 mask.fill_(True);mask[:,::7]=False
 for name,fn in variants.items():
  os.environ['FPF_TRIATT_MASK_STAGE']='off' if name=='baseline' else 'on';kernels=cupti(call,5)
  assert any('triattn_m1_kernel' in k for k in kernels) and not any('flash_triattn' in k for k in kernels),kernels
  assert any('broadcast_mask_stage' in k for k in kernels)==(name=='broadcast'),kernels
  results[name].update(bitwise_equal=True,kernels=kernels)
 for rd in range(3):
  for name in (list(variants) if rd%2==0 else list(reversed(variants))):
   os.environ['FPF_TRIATT_MASK_STAGE']='off' if name=='baseline' else 'on'
   for key,fn in [('core_rounds',core),('block_rounds',call),('stage_rounds',lambda:variants[name](m5,counts))]:results[name][key].append(graph_us(fn))
   print('ROUND',rd,name,results[name],flush=True)
 for r in results.values():r.update(core_us=statistics.median(r['core_rounds']),block_us=statistics.median(r['block_rounds']),stage_us=statistics.median(r['stage_rounds']))
 os.environ.pop('FPF_TRIATT_MASK_STAGE',None)
 out=dict(length=a.length,ending=ending,results=results)
 Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
