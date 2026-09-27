import json,statistics
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm_linear,extension
from miniworld_engine.kernels.layernorm_linear.autograd import layernorm_linear_triton_fn
from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine import settings
settings.configure(autotune_miss_cap=4)
torch.set_num_threads(4);torch.manual_seed(35);torch.backends.cuda.matmul.allow_tf32=False
extension()
def measure(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):fn()
    times=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*100)
    return statistics.median(times)
results=[]
for m,d,n in [(8192,64,64),(147456,128,16),(8192,384,512),(1024,1024,256)]:
    for dt in (torch.bfloat16,torch.float32):
        x=torch.randn(m,d,device='cuda',dtype=dt,requires_grad=True)
        g=torch.randn(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
        w=(torch.randn(n,d,device='cuda',dtype=dt)/d**.5).requires_grad_();bias=torch.randn(n,device='cuda',dtype=dt,requires_grad=True)
        dy=torch.randn(m,n,device='cuda',dtype=dt)
        params=(x,g,b,w,bias)
        def ref():return torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(),(d,),g,b,1e-5).to(dt),w,bias)
        def cud():return cuda_layernorm_linear(*params,threads=128,rows=4)
        def tri():return torch.nn.functional.linear(triton_layernorm(x.unsqueeze(0),g,b,1e-5).squeeze(0),w,bias)
        row={'M':m,'D':d,'N':n,'dtype':str(dt),'times':{}}
        for name,fn in [('torch',ref),('triton_composed',tri),('cuda_composed',cud)]:
            try:
                y=fn();yr=ref();tol=.03 if dt==torch.bfloat16 else 1e-4
                torch.testing.assert_close(y,yr,atol=tol,rtol=tol)
                for ga,gb in zip(torch.autograd.grad(y,params,dy),torch.autograd.grad(yr,params,dy)):
                    rel=(ga.float()-gb.float()).norm()/(gb.float().norm()+1e-8)
                    assert rel<(.03 if dt==torch.bfloat16 else .003),float(rel)
                del y,yr,ga,gb
                def train():return torch.autograd.grad(fn(),params,dy)
                row['times'][name]={'fwd_us':measure(fn),'train_us':measure(train)}
            except Exception as ex:raise RuntimeError((name,m,d,n,dt)) from ex
        results.append(row);print(json.dumps(row),flush=True)
        Path(__file__).with_name('linear-bench.json').write_text(json.dumps(results,indent=2))
