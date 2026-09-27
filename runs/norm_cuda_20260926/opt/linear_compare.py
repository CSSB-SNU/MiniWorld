import json,statistics,gc
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm_linear,extension
from miniworld_engine.kernels.norm_cuda.linear import fused_layernorm_linear,extension as linear_extension
from miniworld_engine.kernels.layernorm_linear.interface import layernorm_linear_triton
from miniworld_engine.kernels.layernorm_linear.cute import layernorm_linear as cute
from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=24)
torch.set_num_threads(4);torch.manual_seed(85);torch.backends.cuda.matmul.allow_tf32=False
extension();linear_extension()
def measure(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):
  for _ in range(10):fn()
 times=[]
 for _ in range(7):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*100)
 return statistics.median(times)
results=[]
for m,d,n in [(8192,64,64),(147456,128,16),(8192,128,128),(8192,256,256),(8192,384,512),(8192,512,128)]:
 for dt in (torch.bfloat16,torch.float16):
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);gamma=torch.randn(d,device='cuda',requires_grad=True);beta=torch.randn_like(gamma,requires_grad=True);w=(torch.randn(n,d,device='cuda',dtype=dt)/d**.5).requires_grad_();b=torch.randn(n,device='cuda',dtype=dt,requires_grad=True);params=x,gamma,beta,w,b;dy=torch.randn(1,m,n,device='cuda',dtype=dt)
  def ref():return torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(),(d,),gamma,beta).to(dt),w,b)
  def fused():return fused_layernorm_linear(*params)
  def composed():return cuda_layernorm_linear(*params)
  def tricomposed():return torch.nn.functional.linear(triton_layernorm(x,gamma,beta,1e-5),w,b)
  row={'M':m,'D':d,'N':n,'dtype':str(dt),'times':{}}
  yr=ref();gr=torch.autograd.grad(yr,params,dy);yr=yr.detach()
  for name,fn in [('native_fused',fused),('native_composed',composed),('triton_composed',tricomposed)]:
   y=fn();gg=torch.autograd.grad(y,params,dy)
   errors=[float((u.float()-v.float()).norm()/v.float().norm().clamp_min(1e-10)) for u,v in zip((y,*gg),(yr,*gr))]
   assert max(errors)<(.02 if dt==torch.bfloat16 else .003),(name,m,d,n,dt,errors)
   del y,gg
   def train():return torch.autograd.grad(fn(),params,dy)
   row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train),'errors':errors}
   with torch.no_grad():row['times'][name]['inference_us']=measure(fn)
  for name,fn in [('triton_fused_forward',lambda:layernorm_linear_triton(*params)),('cute_fused_forward',lambda:cute(x.reshape(-1,d),gamma,beta,w,b).reshape_as(dy))]:
   try:
    with torch.no_grad():
     y=fn();error=float((y.float()-yr.float()).norm()/yr.float().norm());assert error<.03,error
     row['times'][name]={'fwd_us':measure(fn),'error':error}
   except Exception as ex:row['times'][name]={'unavailable':str(ex)}
  results.append(row);print(json.dumps(row),flush=True);Path(__file__).with_name('linear-v5.json').write_text(json.dumps(results,indent=2));del yr,gr,x,gamma,beta,w,b,params,dy;gc.collect();torch.cuda.empty_cache()
