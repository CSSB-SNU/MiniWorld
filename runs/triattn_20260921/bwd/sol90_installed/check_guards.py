"""Installed fusion opt-outs and frontend metadata guards."""
import faulthandler
faulthandler.enable()
import json
import os
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward

torch.manual_seed(93501)
records = []
for ending in (False, True):
    model = TriangleAttention(128, n_head=4, d_hidden=128, starting=not ending,
                              implementation='triton', p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name, w in model.named_parameters():
            if w.ndim >= 2:
                w.normal_(std=w.shape[-1] ** -.5)
    x = torch.randn(1, 384, 384, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    weights = tuple(getattr(model, n).weight for n in ('to_query', 'to_key', 'to_value', 'to_gate', 'to_bias'))
    assert ln_backward.can_use(x, weights, model.ln_pair.weight, model.ln_pair.bias)
    for xx in (x.transpose(1, 2), x[:, :64, :64].contiguous(), x.expand(2, -1, -1, -1).contiguous(), x.float()):
        assert not ln_backward.can_use(xx, weights, model.ln_pair.weight, model.ln_pair.bias)
    def run():
        y = model(x)
        return [y.detach(), *torch.autograd.grad(y, (x, *model.parameters()), dy)]
    ref = run()
    for flag, absent in [('_fuse_front_backward', ('projection_ln_residual_tma',)),
                         ('_fuse_projection_backward', ('projection_ln_residual_tma',)),
                         ('_fuse_dq_backward', ('dq_tma', 'gate_delta_tma')),
                         ('_fuse_gate_backward', ('gate_delta_tma',)),
                         ('_fuse_bias_backward', ('grouped_dkdv', 'dq_tma'))]:
        print('GUARD_START',ending,flag,flush=True)
        setattr(model, flag, False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            got = run()
        names = [e.name for e in p.events()]
        assert all(not any(k in n for n in names) for k in absent), (flag, names)
        assert torch.equal(ref[0], got[0])
        errors = [float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-8)) for a, b in zip(ref, got)]
        assert max(errors) < .015, (flag, errors)
        setattr(model, flag, True)
        records.append(dict(ending=ending, flag=flag, errors=errors))
print('GUARDS_PASS', len(records), flush=True)
(Path(__file__).parent / ('guards-' + os.environ.get('SLURM_JOB_ID', 'local') + '.json')).write_text(json.dumps(records, indent=2) + '\n')
