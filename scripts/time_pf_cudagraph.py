"""MiniPairformer (48 blocks) via a MANUAL CUDA graph — the trunk is static-shape and
sync-free, so it captures cleanly (unlike the full model, blocked by the embedder's
.item()/varlen ops). Gives the launch-overhead-free trunk cost: if this is far below the
reduce-overhead number (33/91ms), the earlier 'slowdown' was per-launch/graph-break
overhead, not real compute — and the modules.* vs ops.* wiring difference vanishes here."""
from __future__ import annotations
import os, statistics
import torch
from team_gm.modules import MiniPairformer
from team_gm.modules.exceptions import ImplementationType

torch.manual_seed(0); dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "256")); D = int(os.environ.get("D_PAIR", "128"))
NB = int(os.environ.get("NB", "48")); ITERS = int(os.environ.get("ITERS", "30"))

model = MiniPairformer(MiniPairformer.Config(
    d_pair=D, p_drop=0.25, n_block=NB,
    implementation=ImplementationType.CUEQUIVARIANCE)).to(dev).to(torch.bfloat16)
print("MiniPairformer NB=%d L=%d D=%d" % (NB, L, D))
pair0 = torch.randn(1, L, L, D, device=dev, dtype=torch.bfloat16)
mask = torch.ones(1, L, device=dev, dtype=torch.bool)

def replay_time(g, iters):
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)

# INFERENCE graph
model.eval()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for _ in range(3):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(pair0, mask)
torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
try:
    g_inf = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.cuda.graph(g_inf):
            _ = model(pair0, mask)
    med, lo, hi = replay_time(g_inf, ITERS)
    print("INFERENCE (trunk cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("INFERENCE capture FAILED:", repr(ex)[:180])

# TRAINING graph (fwd + dummy loss + bwd)
model.train()
pair_in = pair0.clone().requires_grad_(True)
def train_once():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(pair_in, mask)
    out.float().pow(2).mean().backward()
st2 = torch.cuda.Stream(); st2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st2):
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        if pair_in.grad is not None: pair_in.grad = None
        train_once()
torch.cuda.current_stream().wait_stream(st2); torch.cuda.synchronize()
try:
    model.zero_grad(set_to_none=False)
    g_tr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_tr):
        train_once()
    med, lo, hi = replay_time(g_tr, ITERS)
    print("TRAINING  (trunk cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("TRAINING capture FAILED:", repr(ex)[:180])
print("DONE")
