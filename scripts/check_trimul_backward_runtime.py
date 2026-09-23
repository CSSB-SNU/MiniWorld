"""Validate unmodified production dispatch with explicit dropout and cache hits."""
import argparse
import json
from pathlib import Path

import torch
from miniworld_engine.autotune import cache
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bt
from miniworld_engine.modules import BidirectionalTriangleMultiplication
from check_trimul_f567_runtime import capture

p = argparse.ArgumentParser()
p.add_argument('--length', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
L = a.length
production_ln, production_dual = bt.input_ln_residual, bt.input_dual_bwd
original_miss = cache._miss
misses = []


def reject_miss(op, *args, **kwargs):
    if op in ('trimul_input_ln_residual_bwd_triton', 'trimul_input_dual_bwd_triton'):
        misses.append((op, args[:3]))
        raise AssertionError(('production backward cache miss', op, args[:3]))
    return original_miss(op, *args, **kwargs)


def split_dual(g, f, w, v, length):
    result = torch.mm(g, w)
    return result.addmm_(f, v)


cache._miss = reject_miss
torch.manual_seed(117)
model = BidirectionalTriangleMultiplication(128, implementation='triton').cuda().bfloat16().train()
with torch.no_grad():
    for name, w in model.named_parameters():
        if 'ln_' not in name:
            w.normal_(std=128**-.5)
x = torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
mask = torch.ones(1, L, device='cuda', dtype=torch.bool)
mask[:, ::3] = False
dy = torch.randn_like(x)
scale = (torch.rand(1, 1, L, 128, device='cuda') > .25).to(x.dtype) / .75
report = dict(length=L, complete=False, explicit_dropout_errors={})

try:
    for label, dropscale in [('holed_scale', scale), ('zero_scale', torch.zeros_like(scale))]:
        reference = None
        for mode in ('baseline', 'production'):
            if mode == 'baseline':
                bt.input_ln_residual = lambda x, w, b, eps: (bt.triton_layernorm(x, w, b, eps), x)
                bt.input_dual_bwd = split_dual
            else:
                bt.input_ln_residual, bt.input_dual_bwd = production_ln, production_dual
            torch.compiler.reset()
            fn = torch.compile(model._forward_triton, fullgraph=True, dynamic=False,
                               options={'triton.cudagraphs': False})
            model.zero_grad(set_to_none=True)
            x.grad = None
            y = fn(x, mask, dropscale)
            y.backward(dy)
            values = [y.detach(), x.grad] + [w.grad for w in model.parameters()]
            assert all(v is not None and torch.isfinite(v).all() for v in values)
            if reference is None:
                reference = [v.clone() for v in values]
            errors = [float((v.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8))
                      for v, r in zip(values, reference)]
            assert max(errors) < .01, (label, errors)
            if label == 'zero_scale':
                assert torch.equal(y, x) and torch.equal(x.grad, dy)
                assert all(torch.count_nonzero(w.grad) == 0 for w in model.parameters())
            report['explicit_dropout_errors'][label] = errors
            del values, y
        del reference
    # The module globals are now the original, unmodified production choices.
    def step():
        model.zero_grad(set_to_none=False)
        x.grad.zero_()
        y = fn(x, mask, scale)
        y.backward(dy)
        return x.grad
    graph, out = capture(step)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert not misses, misses
    report['complete'] = True
    report['package'] = bt.__file__
    a.output.write_text(json.dumps(report, indent=2))
    print('Verified production dropout/residual, all gradients, strict cache hits and graph replay', L)
finally:
    bt.input_ln_residual, bt.input_dual_bwd = production_ln, production_dual
    cache._miss = original_miss
