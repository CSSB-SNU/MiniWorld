"""Fused LN/bias/KV arithmetic, changed graph inputs, and sanitizer fixtures."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension
from check import capture,rel

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',default='h4kv_front1')
ap.add_argument('--length',required=True,type=int)
ap.add_argument('--output',required=True,type=Path)
ap.add_argument('--native-only',action='store_true')
a=ap.parse_args();L=a.length
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension(a.artifact);front=extension('front8')
torch.manual_seed(92693)
records=[]
for ending in (False,True):
    for dtype in (torch.float32,torch.bfloat16):
        for case in (('random',) if a.native_only else ('random','constant','near_constant')):
            x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
            if case=='constant':x.fill_(1)
            elif case=='near_constant':x.mul_(.01).add_(1)
            gamma=(torch.randn(128,device='cuda')*.1+1).to(dtype)
            beta=(torch.randn(128,device='cuda')*.1).to(dtype)
            wb=torch.randn(4,128,device='cuda',dtype=torch.bfloat16)/128**.5
            wk=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5
            wv=torch.randn_like(wk)/128**.5
            mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
            fn=lambda:ext.front_kv(x,gamma,beta,wb,mask,1e-5,ending,wk,wv)
            outputs=fn();torch.cuda.synchronize()
            assert all(torch.isfinite(t).all() for t in outputs)
            if not a.native_only:
                z,bias=front.front(x,gamma,beta,wb,mask,1e-5,ending)
                assert torch.equal(z,outputs[0]),(ending,dtype,case,'Z',rel(outputs[0],z))
                assert torch.equal(bias,outputs[1]),(ending,dtype,case,'bias')
                for got,w in zip(outputs[2:],(wk,wv)):
                    rows=torch.arange(0,L*L,max(1,L*L//101),device=x.device)
                    ref=F.linear(z.reshape(-1,128)[rows].double(),w.double()).bfloat16()
                    assert rel(got.reshape(-1,128)[rows],ref)<.004
                if case=='random':
                    graph,out,_=capture(fn)
                    x.mul_(.75);gamma.mul_(1.1);beta.add_(.01);wk.add_(.003);wv.sub_(.004);wb.mul_(.7);mask[:,::5]=False
                    graph.replay();torch.cuda.synchronize()
                    eager=fn();assert all(torch.equal(g,e) for g,e in zip(out,eager))
                    z,bias=front.front(x,gamma,beta,wb,mask,1e-5,ending)
                    assert torch.equal(z,out[0]) and torch.equal(bias,out[1])
                    del graph,out,eager
            records.append(dict(ending=ending,affine=str(dtype),case=case))
a.output.write_text(json.dumps(dict(artifact=a.artifact,length=L,native_only=a.native_only,records=records,complete=True),indent=2)+'\n')
print('HEAD4_FRONT_CHECK',L,len(records),flush=True)
