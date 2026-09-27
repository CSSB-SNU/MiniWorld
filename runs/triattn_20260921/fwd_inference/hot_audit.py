"""Exercise both max-free and stable retry paths using runtime counters."""
import argparse
import json
from pathlib import Path
import torch
from native import extension
from check import rel, capture

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',default='hot4')
ap.add_argument('--length',type=int,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args(); L=a.length
ext=extension(a.artifact); stable=extension('resident6')
head_last=hasattr(ext,'bias_head_last') and ext.bias_head_last()
torch.manual_seed(92657)
report=dict(artifact=a.artifact,length=L,records=[])
with torch.inference_mode():
    weights=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/128**.5 for _ in range(4)]
    for case in ('normal','all_masked','large_logits','positive_offset','negative_offset','one_retry_tile'):
        z=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
        b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
        if case=='all_masked': b.fill_(torch.finfo(b.dtype).min)
        elif case=='large_logits': z.mul_(8)
        elif case=='positive_offset': b.add_(1000)
        elif case=='negative_offset': b.sub_(1000)
        elif case=='one_retry_tile': b[:,:,:64].add_(1000)
        counts=torch.zeros((),device='cuda',dtype=torch.int32)
        packed=b.permute(0,2,3,1).contiguous() if head_last else b
        got=ext.forward_audit(z,*weights,packed,counts)
        ref=stable.forward(z,*weights,b)
        torch.cuda.synchronize()
        retries=int(counts)
        assert torch.isfinite(got).all()
        error=rel(got,ref)
        assert error<.008,(case,error)
        assert retries==0 if case=='normal' else retries>0,(case,retries)
        report['records'].append(dict(case=case,retries=retries,relative=error))
    z.normal_();b.normal_(std=.5)
    if head_last: packed.copy_(b.permute(0,2,3,1))
    graph,output,_=capture(lambda:ext.forward_audit(z,*weights,packed,counts))
    for offset in (1000.,0.,-1000.,0.):
        b.normal_(std=.5);b[:,:,:64].add_(offset);counts.zero_()
        if head_last: packed.copy_(b.permute(0,2,3,1))
        graph.replay();torch.cuda.synchronize()
        error=rel(output,stable.forward(z,*weights,b))
        retries=int(counts)
        assert error<.008,(offset,error)
        assert retries>0 if offset else retries==0,(offset,retries)
        report['records'].append(dict(case='changed_graph',offset=offset,retries=retries,relative=error))
report['complete']=True
a.output.write_text(json.dumps(report,indent=2)+'\n')
print('HOT_AUDIT',report,flush=True)
