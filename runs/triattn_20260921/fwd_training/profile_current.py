"""Read-only hardware profile of the installed training attention forward.

Uses the real training module to obtain projected operands, then captures its
existing core. This is measurement scaffolding, not a new Triton kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.triton import main as core

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
L = a.length
torch.manual_seed(92301)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
model = TriangleAttention(128, n_head=4, d_hidden=128, starting=True,
                          implementation='triton', p_drop=0).cuda().bfloat16().train()
with torch.no_grad():
    for name, w in model.named_parameters():
        if w.ndim >= 2:
            w.normal_(std=w.shape[-1] ** -.5)
        elif name.endswith('weight'):
            w.fill_(1)
        else:
            w.zero_()
x = torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
mask = torch.ones(1, L, device='cuda', dtype=torch.bool)
mask[:, ::7] = False
original = core._tri_attn_fwd
captured = []


def observe(*args):
    captured.append(args)
    return original(*args)


core._tri_attn_fwd = observe
try:
    y = model(x, mask)
finally:
    core._tri_attn_fwd = original
assert len(captured) == 1, len(captured)
args = captured[0]
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(5):
        values = original(*args)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    values = original(*args)
assert all(torch.isfinite(t).all() for t in values)
rounds = []
for _ in range(9):
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    begin.record()
    for _ in range(20):
        graph.replay()
    end.record()
    end.synchronize()
    rounds.append(begin.elapsed_time(end) * 1000 / 20)
props = torch.cuda.get_device_properties(0)
report = dict(length=L, regime='actual_training_forward_core', starting=True,
              dtype='bfloat16', heads=4, head_dim=32, mask='every_seventh_key',
              device=props.name, sm_count=props.multi_processor_count,
              shared_memory_per_block_optin=props.shared_memory_per_block_optin,
              torch=str(torch.__version__), source=str(Path(core.__file__).resolve()),
              source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
              operand_shapes=[list(t.shape) for t in args[:4]],
              operand_strides=[list(t.stride()) for t in args[:4]],
              output_strides=[list(t.stride()) for t in values],
              config=str(getattr(core._attn_fwd, 'best_config', None)),
              event_median_us=statistics.median(rounds), event_rounds_us=rounds)
print(json.dumps(report, indent=2), flush=True)
torch.cuda.cudart().cudaProfilerStart()
graph.replay()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
report['complete'] = True
a.output.write_text(json.dumps(report, indent=2) + '\n')
