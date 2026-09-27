"""Verify N40 against its exact unfused-PV source algorithm and time the native boundary."""
import argparse,json
from pathlib import Path
import torch
from native import extension
from check import paired,capture,rel
p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--base',required=True)
p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();L=a.length;base=extension(a.base);new=extension(a.artifact)
report=dict(artifact=a.artifact,base=a.base,length=L,records=[])
torch.manual_seed(92710)
with torch.inference_mode():
    w=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(4)]
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    bias=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)
    for case in ('mixed','dense','one_key','all_masked','late_live','large_logits','positive_offset','negative_offset'):
        z.normal_();bias.normal_(std=.5)
        if case=='mixed':bias[...,::7]=torch.finfo(bias.dtype).min
        elif case in ('all_masked','one_key','late_live'):
            bias.fill_(torch.finfo(bias.dtype).min)
            if case=='one_key':bias[...,-3]=0
            elif case=='late_live':bias[...,-64:]=0
        elif case=='large_logits':z.mul_(8)
        elif case=='positive_offset':bias.add_(1000)
        elif case=='negative_offset':bias.sub_(1000)
        ref=base.forward(z,*w,bias);got=new.forward(z,*w,bias);torch.cuda.synchronize()
        assert torch.equal(ref,got),(case,rel(got,ref))
        report['records'].append(dict(case=case,bitwise=True))
    z.normal_();bias.normal_(std=.5);bias[...,::7]=torch.finfo(bias.dtype).min
    timing,graphs=paired([lambda:base.forward(z,*w,bias),lambda:new.forward(z,*w,bias)],32,40)
    assert torch.equal(graphs[0][1],graphs[1][1])
    report['timing']=timing
    for offset in (0,1000,-1000,0):
        z.normal_();w[0].normal_(std=128**-.5);bias.normal_(std=.5);bias[:,:,:64].add_(offset)
        for graph,_,_ in graphs:graph.replay()
        torch.cuda.synchronize();assert torch.equal(graphs[0][1],graphs[1][1])
    report['changed_graph_bitwise']=True
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
print('N40_PAIR',a.artifact,a.base,L,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
