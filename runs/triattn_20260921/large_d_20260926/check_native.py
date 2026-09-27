"""Independent FP64 adjoints and actual CUDA Graph replay for wide core."""
import argparse
import json
from pathlib import Path
import torch
import candidate

ap=argparse.ArgumentParser();ap.add_argument('--head-dim',type=int,required=True)
ap.add_argument('--length',type=int,default=64);ap.add_argument('--mask',default='mixed')
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--native-only',action='store_true')
args=ap.parse_args();D,L=args.head_dim,args.length
torch.manual_seed(92693)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
candidate.preload(D)
q,k,v=[torch.randn(1,L,L,4*D,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,D).permute(0,3,1,2,4) for _ in range(3)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.3
if args.mask=='mixed':b[...,::7]=torch.finfo(b.dtype).min
elif args.mask=='one_key':b[...,1:]=torch.finfo(b.dtype).min
elif args.mask=='all_masked':b.fill_(torch.finfo(b.dtype).min)
elif args.mask!='none':raise ValueError(args.mask)
dy=torch.randn(1,L,L,4*D,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,D).permute(0,3,1,2,4)


def step():
    o,m=candidate.forward(q,k,v,b)
    grads=candidate.backward(q,k,v,b,m,o,dy)
    return o,m,*grads


values=step();torch.cuda.synchronize()
assert all(torch.isfinite(x).all() for x in values)
report=dict(head_dim=D,length=L,mask=args.mask,records=[],native_only=args.native_only)
if not args.native_only:
    assert L<=128,'independent FP64 is deliberately limited to small dense cases'
    qr,kr,vr,br=[x.detach().double().requires_grad_() for x in (q,k,v,b)]
    scores=torch.einsum('bhijd,bhikd->bhijk',qr,kr)*(D**-.5)+br[:,:,None,:,:]
    prob=torch.softmax(scores,dim=-1)
    prob=torch.where((br>-1e20).any(-1)[:,:,None,:,None],prob,0)
    ref=torch.einsum('bhijk,bhikd->bhijd',prob,vr)
    gradients=torch.autograd.grad(ref,(qr,kr,vr,br),dy.double())
    failed=[]
    for name,got,target in zip(('out','dq','dk','dv','db'),(values[0],*values[2:]),(ref,*gradients)):
        diff=got.double()-target
        norm=float(target.norm())
        error=dict(name=name,relative_l2=float(diff.norm()/target.norm().clamp_min(1e-8)),
                   max_abs=float(diff.abs().max()),reference_norm=norm)
        print('FP64',D,L,args.mask,error,flush=True)
        report['records'].append(error)
        passed=error['max_abs']<2e-4 if norm<1e-8 else error['relative_l2']<(.01 if name=='out' else .02)
        if not passed:failed.append(error)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert not failed,failed
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3): warm=step()
torch.cuda.synchronize();del warm
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph,stream=stream): outputs=step()
for _ in range(3):graph.replay()
torch.cuda.synchronize()
assert all(torch.equal(a,b) for a,b in zip(values,outputs))
with torch.no_grad():
    q.mul_(.7).add_(.1);k.mul_(-.8);v.add_(.25);dy.mul_(-.6)
    # Keep masked entries valid while changing the unmasked bias values.
    b.copy_(torch.where(b>-1e20,b*.5-.2,b))
changed=step();graph.replay();torch.cuda.synchronize()
assert all(torch.equal(a,b) for a,b in zip(changed,outputs))
report['changed_input_graph_bitwise']=True
if not args.native_only:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(3):graph.replay()
        torch.cuda.synchronize()
    names=[e.name for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    for expected in ('wide_training_fwd','wide_dq_resident_alias','wide_grouped_dkdv','wide_delta'):
        assert any(expected in n for n in names),(expected,names)
    report['kernels']=names
report['graph_bitwise']=True;report['complete']=True
args.output.write_text(json.dumps(report,indent=2)+'\n')
print('PASS',D,L,args.mask,flush=True)
