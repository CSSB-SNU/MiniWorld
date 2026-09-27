"""Cold fullgraph, custom-op contract, AMP and partial-gradient fusion checks."""
import argparse
import copy
import json
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward
import qg_module
from qg_candidate import load,baseline

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--stage',action='store_true');ap.add_argument('--installed',action='store_true');a=ap.parse_args()
original=baseline().forward
candidate,qg_module=load(a.artifact,a.stage,a.installed)
report=dict(artifact=a.artifact,staged=a.stage,installed=a.installed,records=[])
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False

def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
def model(ending):
    m=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                        implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for n,w in m.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif n.endswith('weight'):w.fill_(1)
            else:w.zero_()
    return m
def compare(got,ref):
    assert len(got)==len(ref) and all(torch.isfinite(t).all() for t in (*got,*ref))
    e=[float((g.detach().float()-r.detach().float()).norm()/r.detach().float().norm().clamp_min(1e-8)) for g,r in zip(got,ref)]
    assert max(e)<.015,e
    return e
def evaluate(m,x,mask,dy):
    y=m(x,mask)
    inputs=tuple(t for t in (x,*m.parameters()) if t.requires_grad)
    return (y.detach(),*torch.autograd.grad(y,inputs,dy))

torch.manual_seed(92719)
m=model(False);L=384
x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
dy=torch.randn_like(x);mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
ln_backward.forward=candidate
compiled=torch.compile(m,backend='inductor',fullgraph=True)
got=evaluate(compiled,x,mask,dy)
ln_backward.forward=original;ref=evaluate(m,x,mask,dy)
report['records'].append(dict(kind='cold_fullgraph',errors=compare(got,ref)));save()
z=x.detach();w=[m.to_query.weight.detach(),m.to_key.weight.detach(),m.to_value.weight.detach(),m.to_gate.weight.detach()]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)
check=torch.library.opcheck(qg_module.native,(z,*w,b),
    test_utils=('test_schema','test_faketensor','test_aot_dispatch_dynamic'))
report['records'].append(dict(kind='opcheck',result=str(check)));save()
if a.stage or a.installed:
    import os
    assert qg_module.can_use(m,x,mask)
    assert not qg_module.can_use(m,x.float(),mask)
    assert not qg_module.can_use(m,x.half(),mask)
    assert not qg_module.can_use(m,x.transpose(1,2),mask)
    assert not qg_module.can_use(m,x[:,:128,:128].contiguous(),mask[:,:128])
    for opt in ('MINIWORLD_TRIATTN_QG_FWD','MINIWORLD_TRIATTN_Q_FWD','MINIWORLD_TRIATTN_TRAINING_FWD'):
        os.environ[opt]='0';assert not qg_module.can_use(m,x,mask);os.environ[opt]='1'
    for opt in ('_fuse_gate_backward','_fuse_bias_backward','_fuse_dq_backward'):
        setattr(m,opt,False);assert not qg_module.can_use(m,x,mask);setattr(m,opt,True)
    if a.installed:
        ln_backward.forward=candidate
        os.environ['MINIWORLD_TRIATTN_QG_FWD']='0'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            values=evaluate(m,x,mask,dy)
        names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        assert any('qkv_attention_resident_qonly' in n for n in names)
        assert not any('qg_attention_fused' in n for n in names)
        compare(values,ref)
        os.environ['MINIWORLD_TRIATTN_QG_FWD']='1'
    report['records'].append(dict(kind='guards',passed=True));save()
for L in (384,768):
    for ending in (False,True):
        m=model(ending)
        for case in ('amp','mask_none','partial_weights','weights_only'):
            for p in m.parameters():p.requires_grad_(True)
            if case in ('partial_weights','weights_only'):
                for p in m.parameters():p.requires_grad_(False)
                m.to_query.weight.requires_grad_(True);m.to_bias.weight.requires_grad_(True)
            x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=case!='weights_only')
            dy=torch.randn_like(x)
            mask=None if case=='mask_none' else torch.ones(1,L,device='cuda',dtype=torch.bool)
            if mask is not None:mask[:,::7]=False
            outputs=[]
            for fn in (original,candidate):
                ln_backward.forward=fn
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=case=='amp'):
                    outputs.append(evaluate(m,x,mask,dy))
            if case=='amp':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.autocast('cuda',dtype=torch.bfloat16):evaluate(m,x,mask,dy)
                names=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
                assert any('qg_attention_fused' in n for n in names),names
                assert not any('training_fwd_stream' in n or 'qkv_attention_resident_qonly' in n for n in names),names
                report['records'].append(dict(kind='native_dispatch',length=L,ending=ending,kernels=names));save()
            report['records'].append(dict(kind=case,length=L,ending=ending,errors=compare(outputs[1],outputs[0])));save()
            print('VERIFY_QG',case,L,ending,flush=True)
ln_backward.forward=original
report['complete']=True;save()
