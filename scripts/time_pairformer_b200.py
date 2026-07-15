"""Isolated MiniPairformer timing on B200 — everything else (embedder, MSA, head,
recycling) removed. Just token_pair -> N x (bidir trimul + transition) -> token_pair.
Reports per-block ms so any block count is derivable; prints N=4 and N=48 explicitly.
compile+cudagraph, inference and training measured separately."""
from __future__ import annotations
import os, statistics
import torch
from team_gm.modules import MiniPairformer
from team_gm.modules.exceptions import ImplementationType

torch.manual_seed(0)
dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "256"))
D = int(os.environ.get("D_PAIR", "128"))
ITERS = int(os.environ.get("ITERS", "10"))
WARMUP = int(os.environ.get("WARMUP", "5"))


def build(n_block):
    cfg = MiniPairformer.Config(
        d_pair=D, p_drop=0.25, n_block=n_block,
        implementation=ImplementationType.CUEQUIVARIANCE,  # -> TRITON trimul/transition (cute on B200)
    )
    return MiniPairformer(cfg).to(dev).to(torch.bfloat16)


def time_loop(step, warmup, iters, cudagraph):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cudagraph:
            torch.compiler.cudagraph_mark_step_begin()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); step(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def measure(n_block):
    model = build(n_block)
    pair0 = torch.randn(1, L, L, D, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(1, L, device=dev, dtype=torch.bool)
    run = torch.compile(lambda p: model(p, mask), mode="reduce-overhead")

    model.eval()
    def infer():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            run(pair0)
    inf = time_loop(infer, WARMUP, ITERS, cudagraph=False)

    model.train()
    def train():
        torch.compiler.cudagraph_mark_step_begin()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = run(pair0).clone()
        out.float().pow(2).mean().backward()
    trn = time_loop(train, WARMUP, ITERS, cudagraph=False)
    return inf, trn


for N in (4, 48):
    inf, trn = measure(N)
    print("N=%2d blocks  L=%d D=%d :  INFERENCE %.3f ms (%.4f/blk)   TRAINING %.3f ms (%.4f/blk)"
          % (N, L, D, inf, inf / N, trn, trn / N))
print("DONE")
