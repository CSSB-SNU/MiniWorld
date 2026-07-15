"""Apples-to-apples: miniworld modules.bidirectional (what the bench measures, 0.434ms)
vs ops.bidirectional (what the deployed team_gm module calls) — SAME harness, same
weights, training, (1,L,L,d). Isolates whether the deployed-op gap is real or methodology."""
from __future__ import annotations
import os, statistics
import torch
from miniworld_kernels import ops
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.exceptions import ImplementationType

torch.manual_seed(0)
dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "256")); D = int(os.environ.get("D_PAIR", "128"))
ITERS = int(os.environ.get("ITERS", "20")); WARMUP = int(os.environ.get("WARMUP", "8"))

M = BidirectionalTriangleMultiplication(
    d_pair=D, implementation=ImplementationType.MINIWORLD,
).to(dev).to(torch.bfloat16)

pair0 = torch.randn(1, L, L, D, device=dev, dtype=torch.bfloat16)
mask = torch.ones(1, L, device=dev, dtype=torch.bool)

def ops_call(p):
    return ops.bidirectional_triangle_multiplicative_update(
        p, mask,
        norm_in_weight=M.ln_pair.weight, norm_in_bias=M.ln_pair.bias,
        to_left_weight=M.to_left.weight, to_left_gate_weight=M.to_left_gate.weight,
        to_right_weight=M.to_right.weight, to_right_gate_weight=M.to_right_gate.weight,
        norm_out_weight=M.ln_out.weight, norm_out_bias=M.ln_out.bias,
        to_out_weight=M.to_out.weight, to_gate_weight=M.to_gate.weight,
    )

def module_call(p):
    return M(p, mask)

def time_train(fn, tag):
    run = torch.compile(fn, mode="reduce-overhead")
    def step():
        torch.compiler.cudagraph_mark_step_begin()
        if pair0.grad is not None: pair0.grad = None
        M.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = run(pair0).clone()
        out.float().pow(2).mean().backward()
    for _ in range(WARMUP): step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); step(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    med = statistics.median(ts)
    print("%-28s TRAINING %.4f ms" % (tag, med)); return med

# correctness: same weights -> outputs should match (fwd)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    a = module_call(pair0).float(); b = ops_call(pair0).float()
    c = (a.flatten() @ b.flatten() / (a.norm()*b.norm()+1e-9)).item()
print("fwd cos(module, ops) = %.5f" % c)

time_train(module_call, "modules.bidirectional (bench)")
time_train(ops_call, "ops.bidirectional (deployed)")
print("DONE")
