import json
import os
from pathlib import Path
import torch
import accumulate
from build import extension
extension('accumulate',512)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
records=[]
for C in (256,512):
    torch.manual_seed(92781)
    for rows in (64,257):
        gs=[torch.randn(rows,c,device='cuda',dtype=torch.bfloat16) for c in (C,C,C,C,4)]
        ws=[torch.randn(c,C,device='cuda',dtype=torch.bfloat16)*C**-.5 for c in (C,C,C,C,4)]
        def step():return accumulate.dgrad(gs,ws)
        got=step();ref=sum(g.double()@w.double() for g,w in zip(gs,ws))
        rel=float((got.double()-ref).norm()/ref.norm());assert rel<.005,rel
        s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):warm=step()
        torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=s):out=step()
        graph.replay();torch.cuda.synchronize();assert torch.equal(out,got)
        for g in gs:g.mul_(.4).add_(.03)
        for w in ws:w.mul_(-.8)
        changed=step();graph.replay();torch.cuda.synchronize();assert torch.equal(out,changed)
        records.append(dict(width=C,rows=rows,relative_l2_fp64=rel,changed_gradient_weight_graph=True))
        print('PASS',records[-1],flush=True)
Path(__file__).with_name(f'accumulate-check-{os.getenv("SLURM_JOB_ID")}-{os.getenv("CHECK_TOOL","plain")}.json').write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')
