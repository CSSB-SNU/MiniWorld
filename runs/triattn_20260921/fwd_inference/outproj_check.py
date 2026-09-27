"""Independent FP64 and changed-graph check of the fused output/residual boundary."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from native import extension
from check import paired,rel,capture

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True);ap.add_argument('--length',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true')
a=ap.parse_args();L=a.length;ext=extension(a.artifact);front=extension('front8');torch.manual_seed(92701)
report=dict(artifact=a.artifact,length=L,records=[],native_only=a.native_only)
with torch.inference_mode():
    for ending in (False,True):
        for case in (['random'] if a.native_only else ['random','zero_input','zero_weight','zero_residual','large','cancellation']):
            z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
            x=torch.randn_like(z);w=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5
            if case=='zero_input':z.zero_()
            if case=='zero_weight':w.zero_()
            if case=='zero_residual':x.zero_()
            if case=='large':z.mul_(32);x.mul_(16)
            if case=='cancellation':
                t=F.linear(z,w);x.copy_(-(t.transpose(1,2) if ending else t))
            def baseline():return front.post(x,F.linear(z,w),ending)
            def candidate():return ext.forward(z,w,x,ending)
            saved=x.clone();got=candidate();torch.cuda.synchronize()
            assert torch.equal(x,saved) and got.data_ptr()!=x.data_ptr()
            assert got.shape==x.shape and got.is_contiguous() and torch.isfinite(got).all()
            rec=dict(ending=ending,case=case)
            if not a.native_only:
                ref=baseline();rec['relative']=rel(got,ref);rec['max_abs']=float((got.float()-ref.float()).abs().max())
                # Cancellation makes relative-to-output ill-conditioned; normalize by update too.
                scale=F.linear(z,w).float().norm().clamp_min(1e-12)
                rec['update_normalized_error']=float((got.float()-ref.float()).norm()/scale)
                assert rec['relative']<.001 or rec['update_normalized_error']<.0005,rec
                ii=torch.tensor([0,L//2,L-1],device='cuda');jj=torch.tensor([0,1,L//3,L-1],device='cuda')
                zsmall=z[:,ii][:,:,jj]
                exact=F.linear(zsmall.double(),w.double()).bfloat16()
                residual=x.transpose(1,2) if ending else x
                exact=exact+residual[:,ii][:,:,jj]
                def take(t):return (t.transpose(1,2) if ending else t)[:,ii][:,:,jj]
                # Use nonzero input/update normalization even for exact cancellation.
                denom=F.linear(zsmall.double(),w.double()).norm().clamp_min(1e-12)
                rec['fp64_update_error']=float((take(got).double()-exact.double()).norm()/denom)
                rec['baseline_fp64_update_error']=float((take(ref).double()-exact.double()).norm()/denom)
                assert rec['fp64_update_error']<.0005,rec
                assert rec['fp64_update_error']<=rec['baseline_fp64_update_error']*1.15+.0002,rec
                if case=='random':
                    timing,graphs=paired([baseline,candidate],32,40);rec['timing']=timing
                    z.mul_(.7);x.add_(.03);w.sub_(.002)
                    for g,_,_ in graphs:g.replay()
                    torch.cuda.synchronize();rec['changed_graph_relative']=rel(graphs[1][1],graphs[0][1])
                    assert rec['changed_graph_relative']<.001,rec
                    del graphs
                if case in ('zero_input','zero_weight'):assert torch.equal(got,x)
            report['records'].append(rec);a.output.write_text(json.dumps(report,indent=2)+'\n')
            print('OUTPROJ_CHECK',L,rec,flush=True)
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
