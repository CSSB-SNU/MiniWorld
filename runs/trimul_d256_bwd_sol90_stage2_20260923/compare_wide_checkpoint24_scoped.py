from pathlib import Path
import sys,os,json,hashlib
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint24 import Training as WideTraining,configuration
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
import torch._functorch.config as functorch_config
functorch_config.donated_buffer=False
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-checkpoint24-D{D}-L{N}-{record["job"]}.json'
if os.environ.get('COMPARE_TRITON')=='1':path=path.with_name(path.name.replace('wide-checkpoint24-','wide-checkpoint24-vs-triton-'))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p
 newfull=plan
 new=plan.backward
 newfull();torch.cuda.synchronize()
 record['artifacts']=[dict(path=str(a),sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest()) for a in plan.artifacts]
 checks=[]
 for vf in THIS.glob(f'validation-wide-checkpoint24-D{D}-L{N}-*.json'):
  v=json.loads(vf.read_text())
  if v.get('complete') and v['artifacts']==record['artifacts'] and len(v['stress'])==5 and all(c['passed'] for c in v['stress']):checks.append(vf)
 assert checks,'No matching complete strict/graph/PyTorch validation'
 record['validation']=str(max(checks,key=lambda a:a.stat().st_mtime))
 array=os.environ['VALIDATION_ARRAY_JOB'];record['sanitizers']={}
 for tool in ('memcheck','racecheck','synccheck'):
  log=THIS/f'{tool}-wide-checkpoint24-{array}_{index}.log';content=log.read_text()
  expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
  assert expected in content,(tool,content)
  record['sanitizers'][tool]=str(log)
 for branch in (('FORCE_NORM_OVERFLOW','FORCE_ROW_OVERFLOW') if D==512 else ()):
  for tool in ('memcheck','racecheck','synccheck'):
   log=THIS/f'{tool}-wide-checkpoint24-{branch}-{array}_{index}.log';content=log.read_text()
   expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
   assert expected in content,(tool,branch,content)
   record['sanitizers'][branch+'_'+tool]=str(log)
 validation_record=json.loads(Path(record['validation']).read_text())
 record['source_sha256']={name:hashlib.sha256((THIS/name).read_bytes()).hexdigest() for name in validation_record['source_sha256']}
 assert record['source_sha256']==validation_record['source_sha256'],'Validation source changed'
 record['configuration']=configuration(plan)
 assert json.loads(Path(record['validation']).read_text())['configuration']==record['configuration']
 record['qualified']=True
 if record['qualified']:
  functions=dict(our_forward=plan.forward,our_backward=new,new_full=newfull)
  if os.environ.get('COMPARE_TRITON')=='1':
   from miniworld_engine import settings
   settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
   tls=tuple(t.detach().clone().requires_grad_(True) for t in leaves)
   compiled=torch.compile(triton,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
   def baseline_forward():
    with torch.enable_grad():
     return compiled(*tls,mask,ds)
   def baseline():
    with torch.enable_grad():
     y=baseline_forward();return y,torch.autograd.grad(y,tls,dy)
   bs=torch.cuda.Stream();bs.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(bs):
    saved_y=baseline_forward();ty,tg=baseline()
   def baseline_backward():
    with torch.enable_grad():return torch.autograd.grad(saved_y,tls,dy,retain_graph=True)
   torch.cuda.current_stream().wait_stream(bs)
   frozen=[ty.clone(),*[t.clone() for t in tg]];cy,cg=newfull()
   record['triton_errors']={n:error(a,b) for n,a,b in zip(names,[cy,*cg],[ty,*tg])}
   assert all(v<(.005 if n=='y' else .01) for n,v in record['triton_errors'].items()),record['triton_errors']
   import validate_engine as benchmark
   original_capture=benchmark.capture
   def capture_baseline(fn):
    if fn is not baseline and fn is not baseline_backward:return original_capture(fn)
    bs.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(bs):
     for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(bs)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=bs):out=fn()
    graph.replay();torch.cuda.synchronize()
    actual,expected=(out,frozen[1:]) if fn is baseline_backward else ([out[0],*out[1]],frozen)
    ge=[error(a,b) for a,b in zip(actual,expected)]
    assert max(ge)<5e-6,ge
    record['triton_backward_graph_errors' if fn is baseline_backward else 'triton_full_graph_errors']=ge
    return graph,out
   benchmark.capture=capture_baseline
   functions['triton_full']=baseline
   functions['triton_forward']=baseline_forward
   functions['triton_backward']=baseline_backward
   record['baseline']='Existing static-compiled Triton heuristic-24; full CUDA graph replay.'
  # Six simultaneously retained graphs exceed 80GB after preactivation saves.
  # Keep each baseline/candidate pair together, release its private pool, then
  # capture the next pair. F+B is still directly timed, not summed.
  import gc
  record['timing_protocol']='Paired by scope; CUDA graphs released between scopes'
  record['times']={}
  for scope in ('backward','forward','full'):
   our_key='new_full' if scope=='full' else 'our_'+scope
   triton_key='triton_'+scope
   pair={our_key:functions[our_key]}
   if triton_key in functions:pair[triton_key]=functions[triton_key]
   record['times'].update(paired(pair))
   if scope=='backward' and 'triton_full' in functions:saved_y=None
   torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
  if 'triton_full' in functions:
   record['speedup_vs_triton']={scope:record['times']['triton_'+scope]['median_us']/record['times']['new_full' if scope=='full' else 'our_'+scope]['median_us'] for scope in ('forward','backward','full')}
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
