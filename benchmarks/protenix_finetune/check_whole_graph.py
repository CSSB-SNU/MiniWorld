"""Is the whole-step graph right? From one saved state (parameters, Adam state, EMA), with the same seed: eager step vs graph
replay -- loss, gradients, the parameter update -- next to eager vs eager. PFX_DET=1 replaces every random draw by a fixed
pattern."""
import os, sys, runpy
from pathlib import Path
import torch
if os.environ.get("PFX_DET") == "1":
    def _pat(shape, dtype=None, device=None):
        n = 1
        for d in shape: n *= d
        dt = dtype if (dtype is not None and dtype.is_floating_point) else torch.float32
        return (torch.arange(n, device=device, dtype=torch.float32).mul_(0.7071).sin_()).reshape(shape).to(dt)
    def _shape(args, kw):
        if "size" in kw: sz = kw.pop("size")
        else: sz = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else args
        return tuple(sz)
    torch.randn = lambda *a, **kw: _pat(_shape(a, kw), kw.get("dtype"), kw.get("device"))
    torch.rand = lambda *a, **kw: _pat(_shape(a, kw), kw.get("dtype"), kw.get("device")).abs()
    torch.randn_like = lambda t, **kw: _pat(t.shape, kw.get("dtype", t.dtype), t.device)
    torch.rand_like = lambda t, **kw: _pat(t.shape, kw.get("dtype", t.dtype), t.device).abs()
    torch.randperm = lambda n, **kw: torch.arange(n, device=kw.get("device"))
os.environ["PFX_G4_TIME"] = "0"
g4 = runpy.run_path(str(Path(__file__).with_name("whole_graph.py")))
m, G, gr, whole, L = g4["m"], g4["G"], g4["gr"], g4["whole"], g4["L"]
PARAMS, OPT, SHADOW, OPTP = G["PARAMS"], G["OPT"], G["EMA_SHADOW"], G["OPT_PARAMS"]
GREF = [p.grad for p in PARAMS]                      # the graph's gradient buffers
ST = [t for st in OPT.state.values() for t in st.values() if torch.is_tensor(t)] + (G["FOPT"].state_tensors() if G.get("FOPT") else [])
S0 = ([p.detach().clone() for p in PARAMS], [t.clone() for t in ST], [t.clone() for t in SHADOW], [p.detach().clone() for p in OPTP])
def restore():
    with torch.no_grad():
        for d, s in zip(PARAMS, S0[0]): d.copy_(s)
        for d, s in zip(ST, S0[1]): d.copy_(s)
        for d, s in zip(SHADOW, S0[2]): d.copy_(s)
        for d, s in zip(OPTP, S0[3]): d.copy_(s)
    torch.autograd.graph.increment_version(PARAMS)
def flat(ts): return torch.cat([t.detach().float().flatten() for t in ts])
def run(kind, seed):
    restore()
    if kind == "eager":
        for p in PARAMS: p.grad = None
    else:
        for p, gref in zip(PARAMS, GREF): p.grad = gref
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    if kind == "eager":
        l = whole()
    else:
        gr.replay(); l = L
    torch.cuda.synchronize()
    torch.autograd.graph.increment_version(PARAMS)
    out = (l.item(), flat([p.grad for p in PARAMS]), flat(OPTP) - flat(S0[3]), flat(SHADOW) - flat(S0[2]))
    for p, gref in zip(PARAMS, GREF): p.grad = gref
    return out
rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()
e1, e2, g1, g2 = run("eager", 1234), run("eager", 1234), run("graph", 1234), run("graph", 1234)
g3 = run("graph", 99)
print(f"CHECK4 loss eager {e1[0]:.6f} / eager {e2[0]:.6f} / graph {g1[0]:.6f} / graph {g2[0]:.6f} / graph seed 99 {g3[0]:.6f}")
for i, nm in ((1, "grads"), (2, "master update"), (3, "EMA update")):
    print(f"CHECK4 {nm:12s} rel diff: eager/eager {rel(e2[i], e1[i]):.2e} | graph/eager {rel(g1[i], e1[i]):.2e} | graph/graph {rel(g2[i], g1[i]):.2e} | seed99/seed1234 {rel(g3[i], g1[i]):.2e}")
