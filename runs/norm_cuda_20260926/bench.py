import argparse,json,statistics,gc
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm,cuda_layernorm_linear,extension
from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=4)
p=argparse.ArgumentParser();p.add_argument('--part',type=int,default=0);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(83);torch.backends.cuda.matmul.allow_tf32=False
extension();dt=[torch.bfloat16,torch.float32][a.part]

def measure(fn):
    for _ in range(3):fn()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):fn()
    timings=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize();timings.append(start.elapsed_time(end)*1000/10)
    return statistics.median(timings)

results=[]
for m,d in [(8192,64),(147456,128),(147456,384),(8192,1024)]:
    for rms in (False,True):
        x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True);dy=torch.randn_like(x)
        params=[v for v in (x,w,b) if v is not None]
        def reference():
            xf=x.float()
            return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*w).to(dt) if rms else torch.nn.functional.layer_norm(xf,(d,),w,b,1e-5).to(dt)
        def pt_native():
            return torch.nn.functional.rms_norm(x,(d,),w.to(dt),eps=1e-5) if rms else torch.nn.functional.layer_norm(x,(d,),w,b,1e-5)
        def tri():return triton_rmsnorm(x,w) if rms else triton_layernorm(x,w,b,1e-5)
        def candidate(t,r):return (lambda:cuda_rmsnorm(x,w,threads=t,rows=r)) if rms else (lambda:cuda_layernorm(x,w,b,threads=t,rows=r))
        def train(fn):
            def run():
                out=fn();return torch.autograd.grad(out,params,dy)
            return run
        row={'M':m,'D':d,'rms':rms,'dtype':str(dt),'times':{}}
        for name,fn in [('torch_formula',reference),('triton',tri)]+[(f'cuda_t{t}_r{r}',candidate(t,r)) for t in (128,256) for r in (1,4,16,64)]:
            try:
                row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train(fn))}
            except Exception as exc:row['times'][name]={'error':str(exc)}
        results.append(row);print(json.dumps(row),flush=True)
        Path(__file__).with_name(f'bench-{a.part}.json').write_text(json.dumps(results,indent=2))
        del x,w,b,dy,params;gc.collect();torch.cuda.empty_cache()
