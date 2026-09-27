import json
import os
from pathlib import Path
import statistics
import torch
from projection import extension
ROOT=Path(__file__).resolve().parent
report=[]
for C in (256,512):
    ext=extension(C)
    for L in (384,768):
        torch.manual_seed(92729)
        gs=[torch.randn(L*L,c,device='cuda',dtype=torch.bfloat16) for c in (C,C,C,C,4)]
        ws=[torch.randn(c,C,device='cuda',dtype=torch.bfloat16)*C**-.5 for c in (C,C,C,C,4)]
        graphs=[];out=[]
        for tile in (64,128):
            s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):v=ext.dgrad(gs,ws,tile)
            torch.cuda.synchronize();del v
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=s):v=ext.dgrad(gs,ws,tile)
            graphs.append(g);out.append(v)
        for g in graphs:g.replay()
        torch.cuda.synchronize();assert torch.equal(*out)
        values=[[],[]]
        for r in range(20):
            for i in ((0,1) if r%2==0 else (1,0)):
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(10):graphs[i].replay()
                b.record();b.synchronize();values[i].append(a.elapsed_time(b)/10)
        cell=dict(width=C,length=L,bitwise=True,ms={str(t):statistics.median(v) for t,v in zip((64,128),values)})
        print(cell,flush=True);report.append(cell)
        del gs,ws,graphs,out,g,v;torch.cuda.empty_cache()
(ROOT/f'projection-tiles-{os.getenv("SLURM_JOB_ID")}.json').write_text(json.dumps(report,indent=2)+'\n')
