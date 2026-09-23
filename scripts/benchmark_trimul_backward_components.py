"""Measure each backward fusion boundary, including its old intermediate traffic."""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--length', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
# Driver defaults are read at import time.
os.environ.update(MINIWORLD_DRIVER_LENGTH=str(a.length), MINIWORLD_DRIVER_WIDTH='128',
                  MINIWORLD_DRIVER_HEADS='128')

import torch
from miniworld_engine.autotune import cache
from miniworld_engine.kernels.drivers.trimul_backward import dual_operands, ln_operands
from miniworld_engine.kernels.layernorm.triton.main import _ln_bwd
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import (
    input_dual_bwd, input_ln_residual_bwd,
)
from check_trimul_f567_runtime import capture, paired

original_miss = cache._miss
misses = []


def reject_miss(op, *args, **kwargs):
    if op in ('trimul_input_ln_residual_bwd_triton', 'trimul_input_dual_bwd_triton'):
        misses.append((op, args[:3]))
        raise AssertionError(('component backward cache miss', op, args[:3]))
    return original_miss(op, *args, **kwargs)


cache._miss = reject_miss
torch.manual_seed(936)
g, f, w, v, length = dual_operands()
dy, x, weight, mean, rstd, dr, key = ln_operands()
assert length == a.length and g.shape[0] == x.shape[0] == a.length ** 2


def dual_split():
    out = torch.mm(g, w)
    return out.addmm_(f, v)


def ln_split():
    dx, dw, db = _ln_bwd(dy, x, weight, mean, rstd, None, False, list(x.shape), key)
    return dx + dr, dw, db


functions = {
    'B9_B10_split': dual_split,
    'B9_B10_fused': lambda: input_dual_bwd(g, f, w, v, length),
    'B11_B12_split': ln_split,
    'B11_B12_fused': lambda: input_ln_residual_bwd(dy, x, weight, mean, rstd, dr, key),
}
graphs, outputs = {}, {}
for name, fn in functions.items():
    print('CAPTURE', name, 'rows', x.shape[0], flush=True)
    graphs[name], outputs[name] = capture(fn)
    graphs[name].replay()
torch.cuda.synchronize()
errors = {}
for boundary in ('B9_B10', 'B11_B12'):
    old, new = outputs[boundary + '_split'], outputs[boundary + '_fused']
    if isinstance(old, torch.Tensor):
        old, new = (old,), (new,)
    errors[boundary] = [float((u.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8))
                        for u, r in zip(new, old)]
    assert max(errors[boundary]) < .01, errors
assert not misses, misses
report = dict(length=a.length, rows=x.shape[0], complete=True, errors=errors, timings=paired(graphs),
              scope='Eager op boundaries replayed with CUDA graphs; no gradient reset')
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
