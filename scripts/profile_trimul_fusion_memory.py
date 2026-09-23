"""Measure F4--F7 DRAM counters under ncu --profile-from-start off.

Warmup, allocation and autotuning happen before the profiling region. Profiled
calls run in baseline, A, B order; each retains the same training saved tensors.
"""
import argparse
import json
from pathlib import Path
import torch
from benchmark_trimul_fusion_ab import inputs, baseline
import trimul_output_fusion as fusion

p = argparse.ArgumentParser()
p.add_argument('--tuning', type=Path, required=True)
a = p.parse_args()
r = json.loads(a.tuning.read_text())
inp = inputs(r['length'])
for kind, record in r['tuning'].items():
    fusion.SELECTED[fusion.config_key(kind, inp[0], inp[1], inp[4], inp[5])] = record['winner']
fusion.REQUIRE_TUNED = True
for _ in range(3):
    baseline(inp)
    fusion.output_forward(*inp, 'A')
    fusion.output_forward(*inp, 'B')
torch.cuda.synchronize()
keep = []
torch.cuda.cudart().cudaProfilerStart()
for label in ('baseline', 'A', 'B'):
    with torch.cuda.nvtx.range(label):
        keep.append(baseline(inp) if label == 'baseline' else fusion.output_forward(*inp, label))
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
assert all(torch.isfinite(out[0]).all() for out in keep)
