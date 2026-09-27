"""Direct graph timings of validated D128 and saved-products D256 on one GPU."""
from pathlib import Path
import sys,os,json,importlib.util,hashlib
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
sys.path.insert(0,str(ROOT/'.engine-release-2.0.0/src'))
import torch
import miniworld_engine.kernels.trimul_inproj.cuda as cuda_package
# The frozen D128 runner predates the engine's module rename. Resolve only
# its historical modules from the recorded tree; D256 retains release 2.0.
legacy=R.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/cuda'
cuda_package.__path__.append(str(legacy))
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_training as legacy_runtime
legacy_runtime._upstream=lambda: R.parent/'trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
N=int(('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))])
with_baseline=os.environ.get('COMPARE_BASELINE')=='1'
if with_baseline:
 # Backward-only graph replay retains its saved forward tensors. AOT's
 # donated buffers prohibit retain_graph and may overwrite replay inputs.
 import torch._functorch.config as functorch_config
 functorch_config.donated_buffer=False
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False,models={})
prefix='baseline' if with_baseline else 'comparison'
path=R/f'{prefix}-D128-D256-L{N}-{record["job"]}.json'
policy=load('comparison_fixed128',R.parent/'trimul_ln_gradient_20260922/policy.py')
if N==768:policy.F=policy.load('comparison_fixed128_large',R.parent/'trimul_cuda_widths_opt_20260923/fixed128/plan.py')
with torch.no_grad():
 a=policy.Q.setup(N);d128=policy.Fixed(a);reg=policy.P.Regression(a)
 expected=reg();expected=(expected[0].clone(),tuple(t.clone() for t in expected[1]))
 actual=d128();es=policy.H.errors(actual,expected)
 assert all(v['finite'] and v['relative_l2']<=(0 if n=='forward' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()),es
 old=json.loads((R.parent/f'trimul_cuda_widths_opt_20260923/latest-D128-L{N}.json').read_text())
 cubins={name:dict(path=p.k.unit.cubin_path,sha256=hashlib.sha256(Path(p.k.unit.cubin_path).read_bytes()).hexdigest()) for name,p in [('b1',d128.p1),('b7',d128.p7)]}
 for name,v in cubins.items():assert v['sha256']==old['cubins'][name][0]['sha256'],(name,v,old['cubins'][name])
 _,kept=d128.forward()
 record['models']['128']=dict(strict_errors=es,cubins=cubins,dropout_scales=a['d']['ds'].unique().tolist())
 del reg,expected,actual
 # Older experimental modules use generic names; bind D256's dependencies
 # explicitly after D128 has constructed its independent objects.
 PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
 sys.path.insert(0,str(PRE));sys.path.insert(0,str(R.parent/'trimul_backward_wide_20260923'))
 load('plan',R.parent/'trimul_backward_wide_20260923/plan.py')
 load('b7',PRE/'b7.py');load('selected',PRE/'selected.py')
 sys.path.insert(0,str(R));from selected_current import Training,Previous
 sys.path.insert(0,str(R.parent/'trimul_forward_wide_20260923'))
 from validate_engine import paired,error,capture,setup,T
 from shared_candidate import attach
 leaves,dy,mask,ds,*_=setup(256,N)
 with T.native_context(leaves[0].device):
  d256=Training(leaves,mask,ds,dy);record['models']['256']=dict(candidate=attach(d256));d256()
  functions={'D128_forward':d128.forward,'D128_backward':lambda:d128.backward(kept),'D128_full':d128,
             'D256_forward':d256.forward,'D256_backward':d256.backward,'D256_full':d256}
  # This is a timing comparison of already qualified cubins; verify all
  # returned gradients remain finite and graph replay matches eager here.
  for D,m in ((128,d128),(256,d256)):
   y,g=m();expected=[y.clone(),*[t.clone() for t in g]]
   graph,out=capture(m);graph.replay();torch.cuda.synchronize()
   errs=[error(x,e) for x,e in zip([out[0],*out[1]],expected)]
   assert all(bool(t.isfinite().all()) for t in [out[0],*out[1]]) and max(errs)<5e-6,(D,errs)
   record['models'][str(D)]['graph_errors']=errs
  if with_baseline:
   from miniworld_engine import settings
   from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
   settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
   def make_baseline(D,inputs,input_mask,input_ds,upstream,model):
    tls=tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    tm=input_mask.reshape(1,N,N).bfloat16().clone();ts=input_ds.reshape(1,1,N,D).clone()
    def eager(*args):return B.bidirectional_trimul_triton(*args[:-2],1e-5,1e-5,D,mask=args[-2],dropscale=args[-1])
    compiled=torch.compile(eager,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def forward():
     with torch.enable_grad():return compiled(*tls,tm,ts)
    def full():
     y=forward()
     with torch.enable_grad():return y,torch.autograd.grad(y,tls,upstream)
    backward_stream=torch.cuda.Stream();backward_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(backward_stream):saved_y=forward()
    torch.cuda.current_stream().wait_stream(backward_stream)
    def backward():
     with torch.enable_grad():return torch.autograd.grad(saved_y,tls,upstream,retain_graph=True)
    backward._capture_stream=backward_stream
    gy,gg=full();cy,cg=model();errors=policy.H.errors((cy,cg),(gy,gg))
    assert all(v['finite'] and v['relative_l2']<(.005 if k=='forward' else .01) for k,v in errors.items()),(D,errors)
    backward._expected=tuple(t.clone() for t in gg);backward._width=D
    record['models'][str(D)]['triton_errors']=errors
    print('TRITON_VERIFIED',D,N,{k:v['relative_l2'] for k,v in errors.items()},flush=True)
    return dict(forward=forward,backward=backward,full=full)
   for D,inputs,im,ids,upstream,model in ((128,a['d']['leaves'],a['d']['mask'],a['d']['ds'],a['dy'],d128),(256,leaves,mask,ds,dy,d256)):
    baseline=make_baseline(D,inputs,im,ids,upstream,model)
    for scope,fn in baseline.items():functions[f'Triton_D{D}_{scope}']=fn
   record['baseline']='Existing engine Triton, static torch.compile, autotune_miss_cap=24, donated_buffer=False for retained backward graph replay; not an exhaustive retuned Triton baseline.'
   # Autograd backward nodes inherit their forward stream. Capture retained
   # backward on that same stream, not a new stream/default-stream bridge.
   import validate_engine as benchmark
   original_capture=benchmark.capture
   def capture_retained(fn):
    stream=getattr(fn,'_capture_stream',None)
    if stream is None:return original_capture(fn)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    graph.replay();torch.cuda.synchronize()
    errors=[error(g,e) for g,e in zip(out,fn._expected)]
    assert max(errors)<5e-6 and all(bool(g.isfinite().all()) for g in out),(fn._width,errors)
    record['models'][str(fn._width)]['triton_backward_graph_errors']=errors
    return graph,out
   benchmark.capture=capture_retained
  print('VERIFIED',N,{d:v.get('cubins',v.get('candidate')) for d,v in record['models'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
  record['times']=paired(functions)
  record['comparison']={scope:record['times']['D256_'+scope]['median_us']/record['times']['D128_'+scope]['median_us'] for scope in ('forward','backward','full')}
  if with_baseline:
   record['speedup_vs_triton']={str(D):{scope:record['times'][f'Triton_D{D}_{scope}']['median_us']/record['times'][f'D{D}_{scope}']['median_us'] for scope in ('forward','backward','full')} for D in (128,256)}
  record['sol_model']={}
  for D in (128,256):
   M=N*N;flops=66*M*D*D+8*D*N**3;payload=20*M*D+44*D*D+48*D+2*M+2*N*D
   bound=max(flops/989.5e12,payload/3.35e12)*1e6
   record['sol_model'][str(D)]=dict(flops=flops,ideal_bytes=payload,bound_us=bound,pct=100*bound/record['times'][f'D{D}_backward']['median_us'])
  record['sol_note']='Same fixed recomputing-policy comparison model, not measured hardware utilization; D256 saves remove 6*M*D^2 executed backward FLOPs.'
record['complete']=True;path.write_text(json.dumps(record,indent=2))
print('TIMES_US',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
print('RATIOS',record['comparison'],'SOL',record['sol_model'],flush=True)
