"""The first backward in a fresh process uses the compiled native gate boundary."""
import os
import faulthandler
import importlib.util
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import gate_backward

faulthandler.enable()
if os.environ.get('CONTEXT_FROM_PACKAGE','0')!='1':
    trial=Path(__file__).parent.parent/'gate_delta/context_ready'
    name=(trial/'module-name.txt').read_text().strip()
    spec=importlib.util.spec_from_file_location(name,trial/'build/triattn_gate_delta.so')
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    gate_backward._EXT=ext
torch.manual_seed(94333)
model=TriangleAttention(128,n_head=4,d_hidden=128,implementation='triton',p_drop=0).cuda().bfloat16().train()
if os.environ.get('CONTEXT_FROM_PACKAGE','0')=='1':assert model._fuse_gate_backward
model._fuse_gate_backward=True
with torch.no_grad():
    for name,w in model.named_parameters():
        if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
x=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
dy=torch.randn_like(x)
compiled=torch.compile(model,backend='inductor',fullgraph=True)
y=compiled(x)
got=[y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy)]
assert all(torch.isfinite(t).all() for t in got)
y=model(x)
ref=[y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy)]
errors=[float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)) for a,b in zip(got,ref)]
assert max(errors)<.015,errors
print('COLD_COMPILE_PASS',errors,flush=True)
