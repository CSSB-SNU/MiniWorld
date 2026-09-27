import argparse,json,hashlib,statistics,gc
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,extension,resolve_config
from miniworld_engine.kernels._nvcc import ensure_cuda_home,load_extension,host_flags
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=24)
root=Path(__file__).parent
torch.set_num_threads(4);torch.manual_seed(442)
ensure_cuda_home();src=root/'baseline.cu';old=load_extension(name='mw_norm_base_'+hashlib.sha256(src.read_bytes()).hexdigest()[:12],sources=[str(src)],extra_cuda_cflags=[*host_flags(),'-O3','-lineinfo']);extension()
class Baseline(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b,rms,t,r):
  y,mean,inv=old.forward(x,w,b,1e-5,rms,t);ctx.save_for_backward(x,w,mean,inv);ctx.args=b is not None,rms,t,r;return y
 @staticmethod
 def backward(ctx,dy):
  x,w,mean,inv=ctx.saved_tensors;dx,dw,db=old.backward(x,dy.contiguous(),w,mean,inv,*ctx.args)
  return dx,dw if w is not None else None,db if ctx.args[0] else None,None,None,None

def measure(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):
  for _ in range(10):fn()
 out=[]
 for _ in range(7):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();g.replay();en.record();en.synchronize();out.append(st.elapsed_time(en)*100)
 return statistics.median(out)

from miniworld_engine.kernels.norm_cuda import cuda_layernorm_linear
from miniworld_engine.kernels.layernorm_linear.interface import layernorm_linear_triton
results=[]
for m,d,n in [(8192,64,64),(147456,128,16),(8192,384,512),(1024,1024,256)]:
 for dt in (torch.bfloat16,torch.float32):
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);g=torch.randn(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True);w=(torch.randn(n,d,device='cuda',dtype=dt)/d**.5).requires_grad_();bias=torch.randn(n,device='cuda',dtype=dt,requires_grad=True);params=x,g,b,w,bias;dy=torch.randn(1,m,n,device='cuda',dtype=dt)
  def oldfn(t,r):return lambda:torch.nn.functional.linear(Baseline.apply(x,g,b,False,t,r),w,bias)
  def new():return cuda_layernorm_linear(*params)
  def ref():return torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(),(d,),g,b).to(dt),w,bias)
  def tri():return torch.nn.functional.linear(layernorm_kernel(x,g,b,1e-5),w,bias)
  def train(fn):return lambda:torch.autograd.grad(fn(),params,dy)
  v=new();e=ref();vg=torch.autograd.grad(v,params,dy);eg=torch.autograd.grad(e,params,dy)
  errors=[float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)) for a,b in zip((v,*vg),(e,*eg))]
  assert max(errors)<(.003 if dt==torch.bfloat16 else .0003),errors
  del v,e,vg,eg
  _,t,r=min((measure(train(oldfn(t,r))),t,r) for t in (128,256) for r in (1,4,16,64));of=oldfn(t,r)
  row={'M':m,'D':d,'N':n,'dtype':str(dt),'old_config':[t,r],'errors':errors,'times':{}}
  for name,fn in [('old_cuda',of),('new_cuda_auto',new),('engine_composed',tri),('pytorch_reference',ref)]:
   row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train(fn))}
   with torch.no_grad():row['times'][name]['inference_us']=measure(fn)
  with torch.no_grad():row['times']['triton_fused_inference']={'inference_us':measure(lambda:layernorm_linear_triton(*params))}
  results.append(row);print(json.dumps(row),flush=True);(root/'selected-linear.json').write_text(json.dumps(results,indent=2));del x,g,b,w,bias,params,dy;gc.collect();torch.cuda.empty_cache()
