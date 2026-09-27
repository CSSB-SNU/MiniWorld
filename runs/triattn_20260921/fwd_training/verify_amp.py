"""Installed BF16 autocast in the supported BF16 parameter/activation regime."""
import argparse
import json
import os
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
report={'records':[]}
for L in (384,768):
    for ending in (False,True):
        torch.manual_seed(92563)
        model=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                                implementation='triton',p_drop=0).cuda().bfloat16().train()
        with torch.no_grad():
            for name,w in model.named_parameters():
                if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
                elif name.endswith('weight'):w.fill_(1)
                else:w.zero_()
        # The enclosing module requires BF16 activations even inside autocast.
        # FP32 activations select its pre-existing PyTorch reference branch.
        x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn_like(x);mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
        outputs=[]
        for enabled in ('0','1'):
            os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']=enabled
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):y=model(x,mask)
            outputs.append((y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy)))
            names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            assert any(('training_fwd_stream' if enabled=='1' else '_attn_fwd') in n for n in names),names
        ref,got=outputs
        assert all(torch.isfinite(t).all() for t in (*ref,*got))
        errs=[float((g.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for g,r in zip(got,ref)]
        assert max(errs)<.015,errs
        report['records'].append(dict(length=L,ending=ending,errors=errs))
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print('AMP_PASS',L,ending,max(errs),flush=True)
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
