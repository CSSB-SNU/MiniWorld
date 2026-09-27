"""Matched front producers; output-layout conversion is outside every timed graph."""
import argparse
import json
from pathlib import Path
import torch
from native import extension
from check import paired

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
old=extension('front8');new=extension('front8_hlast')
torch.manual_seed(92699)
records=[]
with torch.inference_mode():
    for L in (384,768,1024):
        for ending in (False,True):
            x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
            gamma=torch.randn(128,device='cuda')*.1+1
            beta=torch.randn_like(gamma)*.1
            w=torch.randn(4,128,device='cuda',dtype=torch.bfloat16)/128**.5
            mask=torch.rand(1,L,device='cuda')>.15
            f=lambda:old.front(x,gamma,beta,w,mask,1e-5,ending)
            g=lambda:new.front(x,gamma,beta,w,mask,1e-5,ending)
            u,v=f(),g()
            assert torch.equal(u[0],v[0]) and torch.equal(u[1],v[1].permute(0,3,1,2))
            timing,graphs=paired([f,g],48,40)
            x.mul_(.8);gamma.add_(.03);beta.sub_(.01);w.add_(.002);mask[:,::5]=False
            for graph,_,_ in graphs:graph.replay()
            torch.cuda.synchronize();u,v=[entry[1] for entry in graphs]
            assert torch.equal(u[0],v[0]) and torch.equal(u[1],v[1].permute(0,3,1,2))
            rec=dict(length=L,ending=ending,bitwise=True,changed_graph=True,**timing)
            records.append(rec);print('BIAS_LAYOUT_FRONT',L,ending,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
            del graphs
a.output.write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')
