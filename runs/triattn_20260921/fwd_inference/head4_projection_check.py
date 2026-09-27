"""Matched head-count ablation: identical attention, only KV projection width differs."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from miniworld_engine.modules import TriangleAttention
from native import extension
from candidate import load
from check import paired, rel

ap=argparse.ArgumentParser()
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--head1',default='h1kv_stream2')
ap.add_argument('--head4',default='h4kv_stream2')
a=ap.parse_args()
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
head1=extension(a.head1);head4=extension(a.head4)
f1=load(a.head1);f4=load(a.head4)
report=dict(records=[],head1=a.head1,head4=a.head4,scope='paired head1 vs head4 KV projection and identical complete inference FWD')
def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
for L in (384,768,1024):
    torch.manual_seed(92691)
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    wk=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5
    wv=torch.randn_like(wk)/128**.5
    one=head1.project_kv(z,wk,wv);four=head4.project_kv(z,wk,wv)
    errors=[]
    for index,w in enumerate((wk,wv)):
        ref=F.linear(z.reshape(-1,128)[::max(1,L*L//101)].double(),w.double()).bfloat16()
        got=four[index].reshape(-1,128)[::max(1,L*L//101)]
        error=rel(got,ref);assert error<.004,error
        assert torch.equal(one[index],four[index]),'head projection width changed rounded values'
        errors.append(error)
    timing,graphs=paired([lambda:head1.project_kv(z,wk,wv),lambda:head4.project_kv(z,wk,wv)],48,40)
    z.mul_(.75);wk.add_(.003);wv.sub_(.004)
    for graph,_,_ in graphs:graph.replay()
    torch.cuda.synchronize()
    for i in range(2):assert torch.equal(graphs[0][1][i],graphs[1][1][i])
    for i,w in enumerate((wk,wv)):
        assert rel(graphs[1][1][i],F.linear(z,w))<.004
    del graphs
    report['records'].append(dict(kind='projection',length=L,fp64_relative=errors,**timing));save()
    print('HEAD4_PROJECTION',L,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
    for ending in (False,True):
        model=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                                implementation='triton',p_drop=0.).cuda().bfloat16().eval()
        for name,p in model.named_parameters():
            if p.ndim>=2:p.normal_(std=p.shape[-1]**-.5)
            elif name.endswith('weight'):p.fill_(1)
            else:p.zero_()
        mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
        with torch.inference_mode():
            assert torch.equal(f1(model,z,mask),f4(model,z,mask))
            timing,graphs=paired([lambda:f1(model,z,mask),lambda:f4(model,z,mask)],48,40)
            assert torch.equal(graphs[0][1],graphs[1][1]);del graphs
        report['records'].append(dict(kind='full_fwd',length=L,ending=ending,**timing));save()
        print('HEAD4_FULL_FWD',L,ending,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
report['complete']=True;save()
