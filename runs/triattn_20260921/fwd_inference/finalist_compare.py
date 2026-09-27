"""Same-process paired complete-FWD comparison between two inference candidates."""
import argparse,json
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from candidate import load
from check import paired
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',required=True)
p.add_argument('--length',type=int,default=1024);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();L=a.length
old=load(a.base,out_artifact='outproj_c1');new=load(a.artifact,out_artifact='outproj_c1')
report=dict(base=a.base,artifact=a.artifact,length=L,records=[],rounds=64,replays=40)
with torch.inference_mode():
    for ending in (False,True):
        torch.manual_seed(92638)
        model=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                                implementation='triton',p_drop=.25).cuda().bfloat16().eval()
        for name,w in model.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.normal_(mean=1,std=.05)
            else:w.normal_(std=.05)
            if w.ndim==1:w.copy_(w.bfloat16().float())
        x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
        mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
        t,graphs=paired([lambda:old(model,x,mask),lambda:new(model,x,mask)],64,40)
        assert torch.equal(graphs[0][1],graphs[1][1])
        x.mul_(.8);model.to_query.weight.mul_(.9);mask[:,::5]=False
        for graph,_,_ in graphs:graph.replay()
        torch.cuda.synchronize();assert torch.equal(graphs[0][1],graphs[1][1])
        report['records'].append(dict(ending=ending,bitwise=True,changed_graph_bitwise=True,**t))
        print('FINALIST',a.base,a.artifact,L,ending,{k:v for k,v in t.items() if k!='paired_ratios'},flush=True)
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
