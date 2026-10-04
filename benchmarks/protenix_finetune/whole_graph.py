"""The whole training step -- forward, graph-safe Protenix loss, backward, clip + fused Adam + EMA -- as ONE CUDA graph.
With PFX_SIDE_CONF=1 PFX_SIDE_NOJOIN=1 the mini-rollout + confidence-head branch (its forward, its loss and its backward) is
issued on its own stream and never joined before the backward, so inside the graph it runs concurrently with the diffusion /
trunk path (two backward roots; autograd joins the streams at the end of backward, before the optimizer)."""
import os, sys, runpy, time
from pathlib import Path
os.environ["NREP"] = "0"
sys.argv = ["step.py", sys.argv[1]]
import torch
SIDE = torch.cuda.Stream(); torch.cuda.set_stream(SIDE)
g = runpy.run_path(str(Path(__file__).with_name("step.py")))
step, m = g["step"], g["m"]
G = step.__globals__
assert G["GSL"] is not None and G["FULL"] and G["TRAINER"] == "fast", "needs PFX_FULL=1 PFX_TRAINER=fast PFX_LOSS_GS=1"
if os.environ.get("PFX_COMPILE"):
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    for name in os.environ["PFX_COMPILE"].split(","):
        mod = m
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.forward = torch.compile(mod.forward, dynamic=False)
    t0 = time.time()
    for _ in range(3):
        step(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); print(f"compile warmup (3 steps) {time.time() - t0:.0f} s", flush=True)


def timeit(fn, n, zero=True):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.time(); fn()
        if zero:
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return ts


te = timeit(step, 4)
print(f"eager {sorted(te)[2] * 1e3:.0f} ms", flush=True)
PARAMS, OPT = G["PARAMS"], G["OPT"]
G["_DO_OPT"][0] = False
ALL = [p for p in m.parameters()]


opt_core = G["opt_core"]


def whole():
    if torch.cuda.is_current_stream_capturing():
        # the engine's (data_ptr, _version)-keyed weight packs miss during capture -> the packing kernels are recorded and every
        # replay re-packs the current weights
        torch.autograd.graph.increment_version(ALL)
    loss = G["finish"](G["forward_preds"]())      # loss + backward (two roots with the side branch)
    opt_core()
    return loss


for _ in range(2):                                 # warm the exact captured sequence
    m.zero_grad(set_to_none=True); whole(); torch.autograd.graph.increment_version(PARAMS)
torch.cuda.synchronize()
m.zero_grad(set_to_none=True)                       # grads are allocated by the captured backward (graph pool, fixed addresses)
torch.cuda.empty_cache()
gr = torch.cuda.CUDAGraph()
t0 = time.time()
with torch.cuda.graph(gr, stream=SIDE):
    L = whole()
torch.cuda.synchronize(); print(f"whole step captured in {time.time() - t0:.1f} s", flush=True)
torch.autograd.graph.increment_version(PARAMS)


def gstep():
    gr.replay()
    torch.autograd.graph.increment_version(PARAMS)   # fused Adam does not bump _version (eager users of the engine caches)
    return L


if os.environ.get("PFX_G4_TIME", "1") == "1":
    for _ in range(3):
        gstep()
    torch.cuda.synchronize()
    tg = timeit(gstep, 8, zero=False)
    ls = []
    for _ in range(4):
        gstep(); ls.append(L.item())
    bad = sorted({n.split(".")[0] for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()})
    nog = sum(p.grad is None for p in PARAMS)
    ns = int(os.environ.get("PFX_G4_STRESS", "0"))         # extra replays: a concurrency hang would show here
    for _ in range(ns):
        gstep()
    torch.cuda.synchronize()
    if ns:
        print(f"stress: {ns} more replays finished", flush=True)
    print(f"whole-graph step {sorted(tg)[4] * 1e3:.0f} ms  (eager {sorted(te)[2] * 1e3:.0f})  losses {[round(x, 4) for x in ls]}  nonfinite grads {bad}  "
          f"params without grad {nog}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
