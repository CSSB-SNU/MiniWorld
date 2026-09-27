import argparse
import json
import sys
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.layernorm import dispatch as ln_dispatch
from miniworld_engine.kernels.layernorm.compile_native import _fwd_impl, _bwd_atomic_impl, _bwd_persistent_impl
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine.kernels.layernorm_linear.autograd import layernorm_linear_triton_fn
from miniworld_engine.kernels.layernorm_linear.interface import layernorm_linear_triton

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'triton_audit'))
from audit import measure, inference, PortableShapeAdapter

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--small',action='store_true');a=p.parse_args()
settings.configure(engine_backend='triton',autotune_miss_cap=24)
torch.set_num_threads(4)
rows=[]
cases=[('rms',147456,128,128),('rms',589824,128,128),('rms',147456,384,384),('rms',8192,1024,1024),
                  ('ln',147456,384,384),('ln',589824,384,384),('ln',8192,1024,1024),
                  ('linear',147456,128,16),('linear',8192,384,512),('linear',8192,64,64)]
if a.small:cases=[('rms',8192,64,64),('rms_plain',8192,64,64),('rms_plain',8192,32,32)]
for op,m,d,n in cases:
    torch.manual_seed(224)
    x=torch.randn(1,m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn(d,device='cuda',requires_grad=True)
    b=torch.randn_like(w,requires_grad=True)
    args=(x,) if op=='rms_plain' else ((x,w) if op=='rms' else (x,w,b))
    if op=='ln':
        fn=lambda:layernorm_kernel(x,w,b,1e-5)
        ref=lambda:torch.nn.functional.layer_norm(x.float(),(d,),w,b,1e-5).to(x.dtype)
    elif op.startswith('rms'):
        fn=lambda:triton_rmsnorm(x,None if op=='rms_plain' else w,1e-5)
        def ref():
            xf=x.float()
            return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*(1 if op=='rms_plain' else w)).to(x.dtype)
    else:
        W=(torch.randn(n,d,device='cuda',dtype=x.dtype)/d**0.5).requires_grad_()
        bias=torch.randn(n,device='cuda',dtype=x.dtype,requires_grad=True)
        args=(x,w,b,W,bias)
        fn=(lambda:PortableShapeAdapter.apply(*args)) if a.variant=='baseline' else (lambda:layernorm_linear_triton_fn(*args))
        ref=lambda:torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(),(d,),w,b,1e-5).to(x.dtype),W,bias)
    dy=torch.randn(1,m,n,device='cuda',dtype=x.dtype)
    row=dict(op=op,M=m,D=d,N=n)
    if op=='ln' and a.variant=='candidate':
        _,mean,inv=_fwd_impl(x,w,b,1e-5)
        expected=torch.autograd.grad(ref(),args,dy)
        timings={}
        for name,impl in [('atomic',_bwd_atomic_impl),('persistent',_bwd_persistent_impl)]:
            grad=impl(dy,x,w,mean,inv)
            errs=[float((u.double()-v.double()).norm()/v.double().norm().clamp_min(1e-10)) for u,v in zip(grad,expected)]
            assert max(errs)<0.004,(name,errs)
            timings[name]=measure(lambda:impl(dy,x,w,mean,inv))['median_ms']
        best=min(timings,key=timings.get)
        ln_dispatch.store(x.device,d,ln_dispatch.mbucket(m),best,timings,regime=f'{x.dtype}|{w.dtype}')
        row['backward_path']=best;row['backward_paths_ms']=timings
        del expected,grad,mean,inv
    y=fn();expected=ref();actual=torch.autograd.grad(y,args,dy);eg=torch.autograd.grad(expected,args,dy)
    errors=[float((u.double()-v.double()).norm()/v.double().norm().clamp_min(1e-10)) for u,v in zip((y,*actual),(expected,*eg))]
    assert max(errors)<.005,(op,m,d,errors)
    del y,expected,actual,eg
    row['errors']=errors
    row['train_fwd']=measure(fn)
    row['train']=measure(lambda:torch.autograd.grad(fn(),args,dy))
    inf=(lambda:layernorm_linear_triton(*args)) if op=='linear' else fn
    row['inference']=measure(inference(inf))
    if op=='linear':
        composed=lambda:torch.nn.functional.linear(layernorm_kernel(x,w,b,1e-5),W,bias)
        row['composed_train']=measure(lambda:torch.autograd.grad(composed(),args,dy))
    rows.append(row)
    suffix='-small' if a.small else ''
    (root/f'bench-{a.variant}{suffix}.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(row),flush=True)
