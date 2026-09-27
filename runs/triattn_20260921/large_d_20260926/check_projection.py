import argparse
import json
from pathlib import Path
import torch
from projection import extension
ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true')
args=ap.parse_args();C=args.width;torch.manual_seed(92719)
dy=[torch.randn(256,c,device='cuda',dtype=torch.bfloat16) for c in (C,C,C,C,4)]
w=[torch.randn(c,C,device='cuda',dtype=torch.bfloat16)*C**-.5 for c in (C,C,C,C,4)]
ext=extension(C)
def step():return ext.dgrad(dy,w,64)
got=step();torch.cuda.synchronize()
report=dict(width=C,native_only=args.native_only)
if not args.native_only:
    ref=sum(g.double()@ww.double() for g,ww in zip(dy,w))
    rel=float((got.double()-ref).norm()/ref.norm());assert rel<.003,rel
    report['relative_l2_fp64']=rel
s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):warm=step()
torch.cuda.synchronize();del warm
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph,stream=s):out=step()
graph.replay();torch.cuda.synchronize();assert torch.equal(got,out)
for g in dy:g.mul_(.7).add_(.1)
for ww in w:ww.mul_(-.4).add_(.03)
expected=step();graph.replay();torch.cuda.synchronize();assert torch.equal(expected,out)
report['changed_gradient_weight_graph_bitwise']=True;report['complete']=True
args.output.write_text(json.dumps(report,indent=2)+'\n');print('PASS',report,flush=True)
