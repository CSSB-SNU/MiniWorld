import json,re,statistics
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,extension
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=24)
torch.set_num_threads(4);torch.manual_seed(71);torch.backends.cuda.matmul.allow_tf32=False
root=Path(__file__).parent;extension()

def measure(fn):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(4):fn()
 torch.cuda.current_stream().wait_stream(stream)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):
  for _ in range(10):fn()
 values=[]
 for _ in range(7):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();graph.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*100)
 return statistics.median(values)

results=[]
for path in [root/'bench-0.json',root/'bench-1.json']:
 for row in json.loads(path.read_text()):
  _,key=min((v['train_us'],k) for k,v in row['times'].items() if k.startswith('cuda') and 'train_us' in v)
  threads,rows=map(int,re.findall(r'\d+',key));m,d=row['M'],row['D'];rms=row['rms'];dt=getattr(torch,row['dtype'].split('.')[-1])
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x)
  params=[v for v in (x,w,b) if v is not None]
  def pt():
   xf=x.float()
   return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*w).to(dt) if rms else torch.nn.functional.layer_norm(xf,(d,),w,b,1e-5).to(dt)
  def tri():return triton_rmsnorm(x,w) if rms else triton_layernorm(x,w,b,1e-5)
  def engine():return triton_rmsnorm(x,w) if rms else layernorm_kernel(x,w,b,1e-5)
  def candidate():return cuda_rmsnorm(x,w,threads=threads,rows=rows) if rms else cuda_layernorm(x,w,b,threads=threads,rows=rows)
  out={k:row[k] for k in ('M','D','rms','dtype')};out['config']=key;out['times']={}
  for name,fn in [('pytorch_reference',pt),('triton_default24',tri),('engine_dispatch',engine),('cuda_candidate',candidate)]:
   def train():return torch.autograd.grad(fn(),params,dy)
   out['times'][name]={'fwd_us':measure(fn),'train_us':measure(train)}
  results.append(out);(root/'fair-bench.json').write_text(json.dumps(results,indent=2));print(json.dumps(out),flush=True)
