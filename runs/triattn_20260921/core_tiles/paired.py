from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}').replace('e.name[:40]','e.name')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
results={'baseline':[],'broadcast':[]}
with torch.no_grad():
 for mode in ['off','auto']:
  os.environ['FPF_TRIATT_CORE_BROADCAST']=mode
  for _ in range(5):call()
 torch.cuda.synchronize()
 for rd in range(3):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):
   for mode in (['off','auto'] if rd%2==0 else ['auto','off']):
    os.environ['FPF_TRIATT_CORE_BROADCAST']=mode;call()
  for _ in range(100):graph.replay()
  torch.cuda.synchronize()
  kernels=cupti(graph.replay,50)
  assert not any('flash_triattn' in k for k in kernels)
  for name in results:
   prefix='ta_core_broadcast::triattn_m1_kernel' if name=='broadcast' else 'triattn_m1::triattn_m1_kernel'
   hits=[t for n,t in kernels.items() if prefix in n and 'Traits<0>' in n]
   assert len(hits)==1,(name,kernels)
   results[name].append(hits[0])
  print('PAIRED',rd,results,flush=True)
 out=dict(length=a.length,ending=ending,hot_rounds=results,hot_us={n:statistics.median(v) for n,v in results.items()})
 Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
