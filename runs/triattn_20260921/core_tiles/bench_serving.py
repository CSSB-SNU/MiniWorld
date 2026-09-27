from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}').replace('e.name[:40]','e.name')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from opt_core.kernels import triattn_core_broadcast as bridge
os.environ.pop('FPF_TRIATT_MASK_STAGE',None)
def set_mode(name):os.environ['FPF_TRIATT_CORE_BROADCAST']='off' if name=='baseline' else 'auto'
def graph_profile(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):fn()
 for _ in range(50):g.replay()
 torch.cuda.synchronize()
 return cupti(g.replay,50)
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core():return pf.core_attention(q,k,v,b,m5,core='tier:triattn_native')
 results={n:dict(core_rounds=[],block_rounds=[],hot_rounds=[],safe_rounds=[]) for n in ['baseline','broadcast']}
 for pattern in ['dense','empty','ragged','prefix','one_key']:
  if pattern=='dense':mask.fill_(True)
  elif pattern=='empty':mask.fill_(False)
  elif pattern=='ragged':mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.3)
  elif pattern=='one_key':mask.fill_(False);mask[:,a.length//2]=True
  else:mask.fill_(True);mask[:,:73]=False;mask[:,a.length-127:]=False
  set_mode('baseline');y0=call().clone();c0=core().clone()
  set_mode('broadcast');y1=call().clone();c1=core().clone()
  assert torch.equal(y0,y1) and torch.equal(c0,c1),pattern
  print('PASS installed block/core',pattern,flush=True)
 mask.fill_(True);mask[:,::7]=False
 for name in results:
  set_mode(name);kernels=cupti(call,5)
  expected='ta_core_broadcast::triattn_m1_kernel' if name=='broadcast' else 'triattn_m1::triattn_m1_kernel'
  assert any(expected in k for k in kernels) and not any('flash_triattn' in k for k in kernels),kernels
  assert any('broadcast_mask_stage' in k for k in kernels),kernels
  results[name].update(bitwise_equal=True,kernels=kernels)
 for rd in range(3):
  for name in (list(results) if rd%2==0 else list(reversed(results))):
   set_mode(name)
   results[name]['core_rounds'].append(graph_us(core));results[name]['block_rounds'].append(graph_us(call))
   kernels=graph_profile(call)
   results[name]['hot_rounds'].append(sum(t for n,t in kernels.items() if '::triattn_m1_kernel<' in n and 'Traits<0>' in n))
   results[name]['safe_rounds'].append(sum(t for n,t in kernels.items() if '::triattn_m1_kernel<' in n and 'Traits<1024>' in n))
   print('ROUND',rd,name,results[name]['core_rounds'][-1],results[name]['block_rounds'][-1],results[name]['hot_rounds'][-1],flush=True)
 for r in results.values():
  for kind in ['core','block','hot','safe']:r[kind+'_us']=statistics.median(r[kind+'_rounds'])
 os.environ.pop('FPF_TRIATT_CORE_BROADCAST',None)
 q5=q if q.dim()==5 else q.unsqueeze(0)
 assert bridge.select_core(q5,m5) is not None
 assert bridge.select_core(q5,m5,16) is None
 assert bridge.select_core(q5,m5,trace=torch.empty(1,device=q.device)) is None
 assert bridge.select_core(q5,m5.contiguous()) is None
 assert bridge.select_core(q5[:,:2],m5[:,:2]) is None
 out=dict(length=a.length,ending=ending,results=results)
 Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
