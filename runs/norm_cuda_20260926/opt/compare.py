import argparse,hashlib,json,statistics,gc
from pathlib import Path
import torch
import miniworld_engine.kernels.norm_cuda as norm
from miniworld_engine.kernels._nvcc import load_extension,ensure_cuda_home,host_flags
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=24)
p=argparse.ArgumentParser();p.add_argument('--tag',default='v6');p.add_argument('--part',type=int,default=0);a=p.parse_args()
root=Path(__file__).parent
torch.set_num_threads(4);torch.manual_seed(83)
ensure_cuda_home();src=root/'baseline.cu';old=load_extension(name='mw_norm_base_'+hashlib.sha256(src.read_bytes()).hexdigest()[:12],sources=[str(src)],extra_cuda_cflags=[*host_flags(),'-O3','-lineinfo'])
new=norm.extension();norm.extension=lambda:new

def measure(fn):
 for _ in range(3):fn()
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(10):fn()
 out=[]
 for _ in range(7):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
  st.record();g.replay();en.record();en.synchronize();out.append(st.elapsed_time(en)*1000/10)
 return statistics.median(out)

results=[];dt=[torch.bfloat16,torch.float32][a.part]
for m,d in [(8192,64),(147456,128),(147456,384),(8192,1024),(2048,4096)]:
 for rms in (False,True):
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x);params=[v for v in (x,w,b) if v is not None]
  f=norm.cuda_rmsnorm if rms else norm.cuda_layernorm
  def run(t=128,r=64):return f(x,w,threads=t,rows=r) if rms else f(x,w,b,threads=t,rows=r)
  def train(fn):return lambda:torch.autograd.grad(fn(),params,dy)
  row={'M':m,'D':d,'rms':rms,'dtype':str(dt),'times':{}}
  norm.extension=lambda:old
  row['times']['old_default']={'fwd_us':measure(run),'train_us':measure(train(run))}
  baseline=run();bg=torch.autograd.grad(baseline,params,dy);del baseline
  norm.extension=lambda:new
  for t in (128,256):
   for r in (1,4,16,64):
    fn=lambda:run(t,r);name=f'new_t{t}_r{r}'
    y=fn();gg=torch.autograd.grad(y,params,dy)
    err=[float((u.float()-v.float()).norm()/v.float().norm().clamp_min(1e-10)) for u,v in zip(gg,bg)]
    assert max(err)<(.01 if dt==torch.bfloat16 else .0005),(d,rms,err)
    del y,gg
    row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train(fn)),'grad_rel_l2':err}
  def eng():return triton_rmsnorm(x,w) if rms else layernorm_kernel(x,w,b,1e-5)
  try:row['times']['engine']={'fwd_us':measure(eng),'train_us':measure(train(eng))}
  except Exception as exc:row['times']['engine']={'error':str(exc)}
  results.append(row);print(json.dumps(row),flush=True);(root/f'{a.tag}-{a.part}.json').write_text(json.dumps(results,indent=2))
  del x,w,b,dy,params,bg;gc.collect();torch.cuda.empty_cache()
