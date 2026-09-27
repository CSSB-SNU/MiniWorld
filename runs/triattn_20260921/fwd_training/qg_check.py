"""Projection+attention correctness and paired timing, including QKV training saves."""
import argparse
import json
import math
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F
from native import extension

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--length',required=True,type=int)
ap.add_argument('--output',required=True,type=Path)
ap.add_argument('--native-only',action='store_true')
ap.add_argument('--cases',nargs='+',default=['mixed','dense','one_key','all_masked','late_live','large_logits'])
a=ap.parse_args(); L=a.length
root=Path(__file__).resolve().parent
ext=extension(a.artifact); base=extension('cooperative_head2');qbase=extension('q_only_head4')
torch.manual_seed(925261)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False


def rel(x,y):
    return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-15))


def view(x):
    return x.view(1,L,L,4,32).permute(0,3,1,2,4)


def baseline(z,weights,b):
    gate=F.linear(z,weights[3])
    return (*qbase.forward(z,*weights[:3],b),gate)


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):values=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):values=fn()
    return g,values,stream


report=dict(artifact=a.artifact,length=L,native_only=a.native_only,records=[],
            baseline='frozen17774 Q fusion + separate gate projection',
            build=json.loads((root/a.artifact/'build-ready.json').read_text()),
            baseline_build=json.loads((root/'q_only_head4/build-ready.json').read_text()))
for case in a.cases:
    z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    weights=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/math.sqrt(128) for _ in range(4)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
    valid=torch.ones_like(b,dtype=torch.bool)
    if case=='mixed':valid[...,::7]=False
    elif case in ('one_key','all_masked','late_live'):
        valid.zero_()
        if case=='one_key':valid[...,L-3]=True
        elif case=='late_live':valid[...,L-8:]=True
    elif case=='large_logits':z.mul_(4);b.mul_(4)
    b.masked_fill_(~valid,torch.finfo(b.dtype).min)
    got=ext.forward(z,*weights,b);torch.cuda.synchronize()
    assert len(got)==6 and all(torch.isfinite(x).all() for x in got)
    rec=dict(case=case)
    if not a.native_only:
        ref=baseline(z,weights,b)
        assert all(x.stride()==y.stride() for x,y in zip(got,ref))
        rec['projection_rel']=[rel(x,y) for x,y in zip(got[2:],ref[2:])]
        rec['combined_o_rel']=rel(got[0],ref[0])
        rec['combined_lse_max']=float((got[1]-ref[1]).abs().max())
        own=base.forward(*got[2:5],b)
        rec['attention_only_o_rel']=rel(got[0],own[0])
        rec['attention_only_lse_max']=float((got[1]-own[1]).abs().max())
        assert max(rec['projection_rel'])<.003,rec
        assert rec['combined_o_rel']<.008,rec
        assert rec['attention_only_o_rel']<.0001 and rec['attention_only_lse_max']<.0003,rec
        rows=torch.tensor(sorted(set([0,1,L//2,L-1])),device='cuda')
        queries=torch.arange(0,L,max(1,L//17),device='cuda')
        exact_proj=[F.linear(z[:,rows].double(),w.double()).view(1,len(rows),L,4,32).permute(0,3,1,2,4) for w in weights]
        ce=[rel(x[:,:,rows],y) for x,y in zip((*got[2:5],view(got[5])),exact_proj)]
        be=[rel(x[:,:,rows],y) for x,y in zip((*ref[2:5],view(ref[5])),exact_proj)]
        assert all(x<.003 and x<=y*1.15+.0001 for x,y in zip(ce,be)),(ce,be)
        rec.update(fp64_projection_rel=ce,baseline_fp64_projection_rel=be)
        qr,kr,vr=[x.bfloat16().double() for x in exact_proj[:3]]
        logits=qr[:,:,:,queries]@kr.transpose(-1,-2)/math.sqrt(32)+b[:,:,None,queries].double()
        live=valid[:,:,None,queries];logits=logits.masked_fill(~live,-torch.inf)
        exact=logits.softmax(-1).nan_to_num(0.)@vr
        go=got[0][:,:,rows][:,:,:,queries];ro=ref[0][:,:,rows][:,:,:,queries]
        rec['fp64_o_rel']=rel(go,exact);rec['baseline_fp64_o_rel']=rel(ro,exact)
        assert rec['fp64_o_rel']<.005 and rec['fp64_o_rel']<=rec['baseline_fp64_o_rel']*1.15+.0003,rec
        if case=='all_masked':assert torch.count_nonzero(got[0])==0 and torch.equal(got[1],ref[1])
        if case=='mixed':
            graphs=[capture(lambda:baseline(z,weights,b)),capture(lambda:ext.forward(z,*weights,b))]
            times=[[],[]];ratios=[]
            for rnd in range(12):
                pair={}
                for i in ((0,1) if rnd%2==0 else (1,0)):
                    g=graphs[i][0];g.replay()
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    for _ in range(20):g.replay()
                    end.record();end.synchronize()
                    pair[i]=start.elapsed_time(end)*1000/20;times[i].append(pair[i])
                ratios.append(pair[0]/pair[1])
            rec.update(baseline_us=statistics.median(times[0]),candidate_us=statistics.median(times[1]),
                       speedup=statistics.median(ratios),paired_ratios=ratios)
            z.mul_(.75);weights[0].add_(.003);weights[3].sub_(.004);b[...,::5]=torch.finfo(b.dtype).min
            graphs[0][0].replay();torch.cuda.synchronize()
            snapshots=[x.clone() for x in graphs[0][1]]
            graphs[1][0].replay();torch.cuda.synchronize()
            assert all(rel(x,y)<.008 for x,y in zip(graphs[1][1],snapshots)), 'changed-input graph replay'
            del graphs,snapshots
    report['records'].append(rec)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('QG_CHECK',L,rec,flush=True)
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
