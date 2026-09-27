"""Whole-module PyTorch gradients, frozen branches, dropout and SGD fixtures."""
import argparse
import json
from pathlib import Path
import torch
import candidate

ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();C=args.width;L=64
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
candidate.preload(C//4)
report=dict(width=C,length=L,reference='PyTorch dense BF16 full block',cases=[])


def run(ending,dropout,frozen):
    torch.manual_seed(92711+int(ending))
    models=[candidate.OriginalTriton(C,n_head=4,d_hidden=C,starting=not ending,implementation=backend,p_drop=dropout).cuda().bfloat16().train() for backend in ('pytorch','triton')]
    with torch.no_grad():
        for name,w in models[0].named_parameters():
            if w.ndim>=2:w.normal_(std=C**-.5)
            elif name.endswith('weight'):w.normal_(mean=1,std=.1)
            else:w.normal_(std=.1)
        models[1].load_state_dict(models[0].state_dict())
    candidate.attach(models[1])
    if frozen:
        for model in models:
            for name,w in model.named_parameters():
                if name in ('to_key.weight','to_bias.weight','ln_pair.bias'):w.requires_grad_(False)
    x=torch.randn(1,L,L,C,device='cuda',dtype=torch.bfloat16)*.7+.1
    dy=torch.randn_like(x);mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::5]=False
    result=dict(ending=ending,dropout=dropout,frozen=frozen,steps=[])
    for iteration in range(2):
        if iteration:
            x.mul_(-.6).add_(.2);dy.normal_();mask.logical_not_()
        values=[];names=None
        for i,model in enumerate(models):
            xx=x.clone().requires_grad_(not frozen)
            named=[(name,w) for name,w in model.named_parameters() if w.requires_grad]
            if not frozen:named=[('input',xx),*named]
            names=['output',*[name for name,_ in named]]
            torch.manual_seed(92712+iteration)
            y=model(xx,mask)
            grads=torch.autograd.grad(y,[w for _,w in named],dy)
            values.append((y,*grads))
            assert (y.float()-xx.float()).norm()>0
        errors={}
        for name,ref,got in zip(names,values[0],values[1]):
            assert torch.isfinite(got).all() and torch.isfinite(ref).all()
            rel=float((got.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-8))
            errors[name]=rel
            assert rel<.03,(ending,dropout,frozen,iteration,name,rel)
        # Separate BF16 parameters receive their own measured gradient update.
        for model,out in zip(models,values):
            grads=out[1:] if frozen else out[2:]
            with torch.no_grad():
                for w,g in zip((p for p in model.parameters() if p.requires_grad),grads):w.add_(g,alpha=-1e-4)
        result['steps'].append(errors)
    print('QUALIFIED',C,ending,dropout,frozen,max(max(e.values()) for e in result['steps']),flush=True)
    return result


for ending in (False,True):
    for dropout,frozen in ((0,False),(.1,False),(0,True)):
        report['cases'].append(run(ending,dropout,frozen))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
