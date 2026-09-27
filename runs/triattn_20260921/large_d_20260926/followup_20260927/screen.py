"""FP64 projection check, unchanged-gradient check and paired kernel timings."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import torch
from build import extension
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;sys.path.insert(0,str(BASE))
import projection
ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True);ap.add_argument('--width',type=int,default=512)
ap.add_argument('--check-only',action='store_true');ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();C=args.width;ext=extension(args.kind,C);old=projection.extension(C)
torch.manual_seed(92761)
report=dict(kind=args.kind,width=C,job=os.getenv('SLURM_JOB_ID'),checks=[],measurements=[])
def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
def capture(fn):
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):warm=fn()
    torch.cuda.synchronize();del warm
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s):v=fn()
    g.replay();torch.cuda.synchronize();return g,v
def time(g):
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
    for _ in range(10):g.replay()
    b.record();b.synchronize();return a.elapsed_time(b)/10
for rows in (256,384*384,768*768) if not args.check_only else (256,):
    gs=[torch.randn(rows,c,device='cuda',dtype=torch.bfloat16) for c in (C,C,C,C,4)]
    ws=[torch.randn(c,C,device='cuda',dtype=torch.bfloat16)*C**-.5 for c in (C,C,C,C,4)]
    got=ext.dgrad(gs,ws,64);ref=old.dgrad(gs,ws,64);torch.cuda.synchronize()
    assert torch.equal(got,ref),'Expected same FP32 accumulation order and BF16 output'
    if rows==256:
        target=sum(g.double()@w.double() for g,w in zip(gs,ws))
        err=float((got.double()-target).norm()/target.norm());assert err<.003,err
        report['checks'].append(dict(rows=rows,fp64_relative_l2=err,bitwise=True));save()
    g0,o0=capture(lambda:old.dgrad(gs,ws,64));g1,o1=capture(lambda:ext.dgrad(gs,ws,64))
    gs[0].mul_(.7);ws[2].add_(.01)
    for g in (g0,g1):g.replay()
    torch.cuda.synchronize();assert torch.equal(o0,o1)
    if rows>256:
        values=[[],[]]
        for r in range(20):
            for i in ((0,1) if r%2==0 else (1,0)):values[i].append(time((g0,g1)[i]))
        result=dict(rows=rows,old_ms=statistics.median(values[0]),new_ms=statistics.median(values[1]),rounds=values)
        report['measurements'].append(result);print('TIME',args.kind,C,result['old_ms'],result['new_ms'],flush=True);save()
    del gs,ws,got,ref,g0,g1,o0,o1;torch.cuda.empty_cache()
report['complete']=True;save();print('SCREEN_PASS',args.kind,C,flush=True)
