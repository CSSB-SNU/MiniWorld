"""Time bidir trimul and transition SEPARATELY on B200, via team_gm modules with
implementation=TRITON (routes to miniworld_kernels ops / cute on B200). Same block
shape (1,L,L,d) and regime (compile+cudagraph) as the isolated pairformer run, so the
two should roughly sum to the per-block cost."""
from __future__ import annotations
import os, statistics
import torch
from team_gm.modules.layers import BidirectionalTriangleMultiplication, Transition
from team_gm.modules.exceptions import ImplementationType

torch.manual_seed(0)
dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "256"))
D = int(os.environ.get("D_PAIR", "128"))
ITERS = int(os.environ.get("ITERS", "20"))
WARMUP = int(os.environ.get("WARMUP", "8"))


def time_loop(step, warmup, iters):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); step(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def measure(name, mod, needs_mask):
    mod = mod.to(dev).to(torch.bfloat16)
    pair0 = torch.randn(1, L, L, D, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(1, L, device=dev, dtype=torch.bool)
    fn = (lambda p: mod(p, mask)) if needs_mask else (lambda p: mod(p))
    run = torch.compile(fn, mode="reduce-overhead")

    mod.eval()
    def infer():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            run(pair0)
    inf = time_loop(infer, WARMUP, ITERS)

    mod.train()
    def train():
        torch.compiler.cudagraph_mark_step_begin()
        mod.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = run(pair0).clone()
        out.float().pow(2).mean().backward()
    trn = time_loop(train, WARMUP, ITERS)
    print("%-18s L=%d D=%d :  INFERENCE %.4f ms   TRAINING %.4f ms" % (name, L, D, inf, trn))
    return inf, trn


tri = BidirectionalTriangleMultiplication(d_pair=D, implementation=ImplementationType.TRITON)
trs = Transition(D, implementation=ImplementationType.TRITON)
i1, t1 = measure("bidir_trimul", tri, needs_mask=True)
i2, t2 = measure("transition", trs, needs_mask=False)
print("SUM               L=%d D=%d :  INFERENCE %.4f ms   TRAINING %.4f ms" % (L, D, i1 + i2, t1 + t2))
print("DONE")
