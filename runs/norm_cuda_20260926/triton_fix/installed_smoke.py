import hashlib
import json
import sys
from pathlib import Path

import torch
import miniworld_engine
from miniworld_engine import settings
from miniworld_engine.kernels.layernorm import dispatch
from miniworld_engine.kernels.layernorm.compile_native import _resolve_bwd_path
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm
from miniworld_engine.kernels.layernorm_linear.autograd import layernorm_linear_triton_fn

root = Path(__file__).resolve().parent
package = Path(miniworld_engine.__file__).resolve().parent
assert package == Path('/home/psk6950/MiniWorld/.engine-release-2.0.0/src/miniworld_engine')
for rel, expected in json.loads((root / 'installed-manifest.json').read_text()).items():
    assert hashlib.sha256((package / rel).read_bytes()).hexdigest() == expected, rel
sys.path.insert(0, str(root.parent / 'triton_audit'))
from audit import measure

settings.configure(engine_backend='triton', autotune_miss_cap=24)
torch.set_num_threads(4)
torch.manual_seed(224)
results = []
for op, m, d, n in [('rms',8192,1024,1024), ('ln',8192,1024,1024), ('linear',147456,128,16)]:
    x = torch.randn(1,m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w = torch.randn(d,device='cuda',requires_grad=True)
    b = torch.randn(d,device='cuda',requires_grad=True)
    if op == 'rms':
        args = (x,w)
        fn = lambda: triton_rmsnorm(x,w,1e-5)
    elif op == 'ln':
        args = (x,w,b)
        stats = torch.empty(m,device='cuda')
        assert dispatch.lookup(x.device,d,dispatch.mbucket(m),regime=f'{x.dtype}|{w.dtype}') == 'atomic'
        assert _resolve_bwd_path(m,d,x,x,w,stats,stats) == 'atomic'
        fn = lambda: layernorm_kernel(x,w,b,1e-5)
    else:
        W = (torch.randn(n,d,device='cuda',dtype=x.dtype)/d**0.5).requires_grad_()
        bias = torch.randn(n,device='cuda',dtype=x.dtype,requires_grad=True)
        args = (x,w,b,W,bias)
        fn = lambda: layernorm_linear_triton_fn(*args)
    dy = torch.randn(1,m,n,device='cuda',dtype=x.dtype)
    timing = measure(lambda: torch.autograd.grad(fn(),args,dy))
    results.append(dict(op=op,M=m,D=d,N=n,train=timing))
    print(results[-1],flush=True)
(root / 'installed-smoke.json').write_text(json.dumps(dict(package=str(package),results=results),indent=2))
