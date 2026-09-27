"""Check inference-only LN+bias and residual against independent FP64 arithmetic."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension
from check import rel, capture

ap = argparse.ArgumentParser()
ap.add_argument('--length',type=int,default=384)
ap.add_argument('--artifact',default='front')
ap.add_argument('--native-only',action='store_true')
ap.add_argument('--output',type=Path,required=True)
a = ap.parse_args(); L = a.length
torch.set_grad_enabled(False);torch.manual_seed(92688)
ext = extension(a.artifact)
head_last=hasattr(ext,'bias_head_last') and ext.bias_head_last()
report = dict(length=L,artifact=a.artifact,records=[],native_only=a.native_only)
for ending in (False,True):
    for dtype in (torch.float32,torch.bfloat16):
        x = torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
        gamma = torch.randn(128,device='cuda',dtype=dtype)*.1+1
        beta = torch.randn_like(gamma)*.2
        wb = torch.randn(4,128,device='cuda',dtype=torch.bfloat16)/128**.5
        mask = torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
        for case in (['random'] if a.native_only else ['random','constant','near_constant']):
            if case == 'constant': x.fill_(1)
            elif case == 'near_constant': x.normal_(mean=1,std=.008)
            z,b = ext.front(x,gamma,beta,wb,mask,1e-5,ending)
            o = torch.randn_like(x);y = ext.post(x,o,ending);torch.cuda.synchronize()
            rec = dict(ending=ending,dtype=str(dtype),case=case)
            assert torch.isfinite(z).all() and torch.isfinite(b).all() and torch.isfinite(y).all()
            if not a.native_only:
                xx = x.transpose(1,2).contiguous() if ending else x
                exact = F.layer_norm(xx.double(),(128,),gamma.double(),beta.double(),1e-5).bfloat16()
                rec['z_relative'] = rel(z,exact)
                assert rec['z_relative'] < .0003,rec
                exact_b = F.linear(z.double(),wb.double()).bfloat16()
                if not head_last: exact_b=exact_b.permute(0,3,1,2).contiguous()
                valid = (mask[:,None,:,None] if head_last else mask[:,None,None,:]).expand_as(b)
                rec['bias_relative'] = rel(b[valid],exact_b[valid])
                assert rec['bias_relative'] < .0003,rec
                assert torch.all(b[~valid]==torch.finfo(b.dtype).min)
                assert torch.equal(y,x+(o.transpose(1,2).contiguous() if ending else o))
                if case == 'random':
                    graph,out,stream = capture(lambda:ext.front(x,gamma,beta,wb,mask,1e-5,ending))
                    x.mul_(.8);gamma.add_(.03);beta.sub_(.01);wb.add_(.005);mask[:,::5]=False
                    graph.replay();torch.cuda.synchronize()
                    ref = ext.front(x,gamma,beta,wb,mask,1e-5,ending)
                    assert all(torch.equal(u,v) for u,v in zip(out,ref))
                    del graph
            report['records'].append(rec)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print('FRONT_CHECK',rec,flush=True)
report['complete'] = True;a.output.write_text(json.dumps(report,indent=2)+'\n')
