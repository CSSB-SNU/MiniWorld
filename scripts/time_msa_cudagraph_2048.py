"""Isolated 4-block MiniMSAModule (OuterProductMean + bidir trimul + transition) at L=384
via a MANUAL CUDA graph — single pass (no recycle). Grabs the real msa_module inputs with
a forward pre-hook during one eager model forward, then captures msa_module alone."""
from __future__ import annotations
import os, statistics, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path("scripts").resolve()))
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0); dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "384")); ITERS = int(os.environ.get("ITERS", "30"))
cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True); m["trunk"]["pairformer"]["n_block"] = 48
model = MiniSWAModel(MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"])).to(dev)
model._forced_n_recycle = 1
print("msa_module n_block=%d L=%d" % (len(model.msa_module.blocks), L))
batch = _build_precompile_batch(device=dev, msa_depth=2048, n_tokens=L, n_atoms=L*4,
                                n_templates=1, num_res_class=int(m["shared"]["num_res_class"]))

# grab real msa_module inputs via a pre-hook on one eager forward
grabbed = {}
def _hook(mod, args):
    grabbed["args"] = tuple(a.clone().detach() if torch.is_tensor(a) else a for a in args)
h = model.msa_module.register_forward_pre_hook(_hook)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model.forward(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                  sequence=batch.sequence, structure=batch.structure)
h.remove()
args = list(grabbed["args"])
print("grabbed msa_module inputs:", [tuple(a.shape) if torch.is_tensor(a) else a for a in args])

def replay_time(g, iters):
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)

msa = model.msa_module
# INFERENCE
msa.eval()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for _ in range(3):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            msa(*args)
torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
try:
    g_inf = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.cuda.graph(g_inf):
            _ = msa(*args)
    med, lo, hi = replay_time(g_inf, ITERS)
    print("INFERENCE (msa cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("INFERENCE capture FAILED:", repr(ex)[:160])

# TRAINING: pair (args[2]) requires grad
msa.train()
targs = list(args); targs[2] = args[2].clone().requires_grad_(True)
def train_once():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = msa(*targs)
    out.float().pow(2).mean().backward()
st2 = torch.cuda.Stream(); st2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st2):
    for _ in range(3):
        msa.zero_grad(set_to_none=True)
        if targs[2].grad is not None: targs[2].grad = None
        train_once()
torch.cuda.current_stream().wait_stream(st2); torch.cuda.synchronize()
try:
    msa.zero_grad(set_to_none=False)
    g_tr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_tr):
        train_once()
    med, lo, hi = replay_time(g_tr, ITERS)
    print("TRAINING  (msa cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("TRAINING capture FAILED:", repr(ex)[:160])
print("DONE")
