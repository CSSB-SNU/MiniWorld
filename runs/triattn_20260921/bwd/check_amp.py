"""Actual native dispatch under BF16 autocast; FP16 dispatch is declined.

The stock module's FP16 autocast backward already fails in gated projection
(fp16 dO, bf16 W). That separate limitation was observed in job15540.
"""
import copy,json
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention import cuda as native
torch.manual_seed(983322)
result=[]
for dtype in (torch.bfloat16,torch.float16):
    base=TriangleAttention(128,n_head=4,d_hidden=128,implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():base.to_out.weight.normal_(std=.1)
    cand=copy.deepcopy(base);base._fuse_projection_backward=False
    x=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(x);mask=torch.ones(1,384,device='cuda',dtype=torch.bool);mask[:,::7]=False
    if dtype==torch.float16:
        weights=tuple(getattr(cand,n).weight for n in ('to_query','to_key','to_value','to_gate','to_bias'))
        with torch.autocast('cuda',dtype=dtype):assert not native.can_use(x,weights)
        result.append(dict(dtype=str(dtype),native_declined=True,stock_module_supported=False,baseline_failure_job=15540))
        print('FP16_NATIVE_DECLINED',flush=True)
        continue
    refs=[]
    for model in (base,cand):
        with torch.autocast('cuda',dtype=dtype):y=model(x,mask)
        refs.append([y.detach(),*torch.autograd.grad(y,(x,*model.parameters()),dy)])
    errors=[float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)) for a,b in zip(*refs)]
    assert max(errors)<.015,errors
    compiled=torch.compile(cand,backend='inductor',fullgraph=True)
    for _ in range(2):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            with torch.autocast('cuda',dtype=dtype):y=compiled(x,mask)
            values=[y.detach(),*torch.autograd.grad(y,(x,*cand.parameters()),dy)]
    used=any('projection_dgrad_tma' in e.name for e in p.events())
    assert used==(dtype==torch.bfloat16),(dtype,used)
    ce=[float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)) for a,b in zip(refs[1],values)]
    assert max(ce)<.015,ce
    result.append(dict(dtype=str(dtype),relative_l2=errors,compiled_relative_l2=ce,compiled_native=used))
    print('AMP_PASS',dtype,used,flush=True)
(Path(__file__).parent/'amp-correctness.json').write_text(json.dumps(result,indent=2)+'\n')
