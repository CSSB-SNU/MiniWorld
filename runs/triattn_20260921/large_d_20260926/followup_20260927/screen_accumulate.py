import json
import os
from pathlib import Path
import statistics
import sys
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
import projection
import accumulate
from build import extension
extension('accumulate',512)
report=[]
def cap(fn):
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):v=fn()
    torch.cuda.synchronize();del v
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s):v=fn()
    g.replay();torch.cuda.synchronize();return g,v
def time(g):
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
    for _ in range(10):g.replay()
    b.record();b.synchronize();return a.elapsed_time(b)/10
for C in (256,512):
    ext=projection.extension(C)
    for rows in (256,384*384,768*768):
        torch.manual_seed(92771)
        gs=[torch.randn(rows,c,device='cuda',dtype=torch.bfloat16) for c in (C,C,C,C,4)]
        ws=[torch.randn(c,C,device='cuda',dtype=torch.bfloat16)*C**-.5 for c in (C,C,C,C,4)]
        def split():
            dx=gs[0]@ws[0]
            for g,w in zip(gs[1:],ws[1:]):dx=dx+g@w
            return dx
        fns=(split,lambda:ext.dgrad(gs,ws,64),lambda:accumulate.dgrad(gs,ws))
        graphs=[];out=[]
        for fn in fns:
            g,v=cap(fn);graphs.append(g);out.append(v)
        errors={}
        for name,v in zip(('split','prior_fused','beta_accumulate'),out):
            ref=sum(g.double()@w.double() for g,w in zip(gs,ws)) if rows==256 else out[0].float()
            rel=float((v.double()-ref).norm()/ref.norm());assert rel<.01,(name,rel)
            errors[name]=rel
        gs[0].mul_(.7);ws[2].add_(.01)
        expected=accumulate.dgrad(gs,ws);graphs[2].replay();torch.cuda.synchronize()
        assert torch.equal(expected,out[2])
        for g in graphs:g.replay()
        values=[[],[],[]]
        for r in range(18):
            for i in ((0,1,2) if r%2==0 else (2,1,0)):values[i].append(time(graphs[i]))
        record=dict(width=C,rows=rows,errors=errors,ms={n:statistics.median(v) for n,v in zip(('split','prior_fused','beta_accumulate'),values)})
        report.append(record);print(record,flush=True)
        del gs,ws,graphs,out,expected,g,v,ref;torch.cuda.empty_cache()
(ROOT/f'screen-accumulate-{os.getenv("SLURM_JOB_ID")}.json').write_text(json.dumps(report,indent=2)+'\n')
