"""Fresh-process checks of the actual installed training-forward dispatch."""
import argparse
import json
import os
from pathlib import Path

import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import training_forward as native
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
torch.manual_seed(92553)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
report={'records':[]}


def save():a.output.write_text(json.dumps(report,indent=2)+'\n')


def make_model(ending):
    model=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                            implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name,w in model.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    return model


def errors(got,ref):
    assert all(torch.isfinite(t).all() for t in (*got,*ref))
    e=[float((g.detach().float()-r.detach().float()).norm()/r.detach().float().norm().clamp_min(1e-8)) for g,r in zip(got,ref)]
    assert max(e)<.015,e
    return e


# Compile before eager execution warms the new artifact or device caches.
os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='1'
model=make_model(False)
x=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
mask=torch.ones(1,384,device='cuda',dtype=torch.bool);mask[:,::7]=False
dy=torch.randn_like(x)
compiled=torch.compile(model,backend='inductor',fullgraph=True)
y=compiled(x,mask); got=(y,*torch.autograd.grad(y,(x,*model.parameters()),dy))
y=model(x,mask); ref=(y,*torch.autograd.grad(y,(x,*model.parameters()),dy))
report['records'].append(dict(kind='cold_fullgraph',errors=errors(got,ref)));save()
assert native._artifact_ready()

for L in (384,768,1024):
    q,k,v=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
           .view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(3)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)
    assert native.can_use(q,k,v,b)
    got=core._tri_attn_fwd(q,k,v,b,token_key(L))
    ref=native.forward(q,k,v,b)
    assert all(torch.equal(g,r) for g,r in zip(got,ref)), 'installed path differs from qualified artifact'
    if L==384:
        check=torch.library.opcheck(core._tri_attn_fwd,(q,k,v,b,token_key(L)),
            test_utils=('test_schema','test_faketensor','test_aot_dispatch_dynamic'))
        report['records'].append(dict(kind='opcheck',result=str(check)));save()
        assert not native.can_use(q.float(),k.float(),v.float(),b.float())
        assert not native.can_use(q.half(),k.half(),v.half(),b.half())
        assert not native.can_use(q,k,v,b.transpose(-1,-2))
        assert not native.can_use(q.contiguous(),k.contiguous(),v.contiguous(),b)
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='0'
        assert not native.can_use(q,k,v,b)
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='1'
        report['records'].append(dict(kind='guards',float32=True,float16=True,bias_strides=True,qkv_layout=True,opt_out=True));save()
    for ending in (False,True):
        model=make_model(ending)
        x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn_like(x);mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='0'
        y=model(x,mask); ref=(y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy))
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='1'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            y=model(x,mask)
        got=(y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy))
        names=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
        assert any('training_fwd_stream' in n for n in names),names
        assert not any('_attn_fwd' in n for n in names),names
        report['records'].append(dict(kind='installed_dispatch',length=L,ending=ending,
                                      errors=errors(got,ref),kernels=names));save()
report['complete']=True;save()
print('INSTALLED_VERIFIED',flush=True)
