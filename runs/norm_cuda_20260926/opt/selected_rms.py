import argparse,json,hashlib,statistics,gc
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,extension,resolve_config
from miniworld_engine.kernels._nvcc import ensure_cuda_home,load_extension,host_flags
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=24)
p=argparse.ArgumentParser();p.add_argument('--part',type=int,default=0);a=p.parse_args();root=Path(__file__).parent
torch.set_num_threads(4);torch.manual_seed(442);dt=[torch.bfloat16,torch.float32,torch.float16,torch.float64][a.part]
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
results=[]
for m,d in [(147456,128)]:
 for rms in (True,):
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);acc=torch.float64 if dt==torch.float64 else torch.float32;w=torch.randn(d,device='cuda',dtype=acc,requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x);params=tuple(v for v in (x,w,b) if v is not None)
  def oldfn(t,r):return lambda:Baseline.apply(x,w,b,rms,t,r)
  def new():return cuda_rmsnorm(x,w) if rms else cuda_layernorm(x,w,b)
  def ref():
   xf=x.to(acc);return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*w).to(dt) if rms else torch.nn.functional.layer_norm(xf,(d,),w,b).to(dt)
  def train(fn):return lambda:torch.autograd.grad(fn(),params,dy)
  out=new();exp=ref();grad=torch.autograd.grad(out,params,dy);eg=torch.autograd.grad(exp,params,dy)
  errors=[float((v.double()-e.double()).norm()/e.double().norm().clamp_min(1e-12)) for v,e in zip((out,*grad),(exp,*eg))]
  assert max(errors)<{torch.bfloat16:.003,torch.float16:.001,torch.float32:.0003,torch.float64:1e-10}[dt],(d,rms,errors)
  del out,exp,grad,eg
  trials=[(measure(train(oldfn(t,r))),t,r) for t in (128,256) for r in (1,4,16,64)]
  _,t,r=min(trials);of=oldfn(t,r)
  row={'M':m,'D':d,'rms':rms,'dtype':str(dt),'old_config':[t,r],'new_config':resolve_config(x,rms),'errors':errors,'times':{}}
  # Re-measure selected old and new back-to-back; old is tuned, not a poor default.
  for name,fn in [('old_cuda',of),('new_cuda',new),('pytorch_reference',ref)]:
   row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train(fn))}
  if d<4096 and dt!=torch.float64:
   def eng():return triton_rmsnorm(x,w) if rms else layernorm_kernel(x,w,b,1e-5)
   row['times']['engine_dispatch']={'fwd_us':measure(eng),'train_us':measure(train(eng))}
  else:row['engine_unavailable']='autotune key width limit' if d>=4096 else 'FP64 accumulation contract differs'
  results.append(row);print(json.dumps(row),flush=True);(root/f'selected-rms-{a.part}.json').write_text(json.dumps(results,indent=2));del x,w,b,dy,params;gc.collect();torch.cuda.empty_cache()
