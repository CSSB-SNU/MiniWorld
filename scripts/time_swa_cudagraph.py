"""B200 full-model latency via a MANUAL CUDA graph capturing the WHOLE model as ONE
graph (no torch.compile graph-breaks) — inference (fwd) and training (fwd+loss+bwd)
captured & replayed separately. This is the deployment-representative number: every
kernel (FA4 embedder, cute/triton trunk, pointwise) is in a single graph, so per-launch
host overhead -> 0 (unlike reduce-overhead, which fragments at each @compiler.disable)."""
from __future__ import annotations
import os, statistics, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path("scripts").resolve()))
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0); dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "256")); N_ATOM = int(os.environ.get("N_ATOM", str(L*4)))
N_REC = int(os.environ.get("N_REC", "4")); ITERS = int(os.environ.get("ITERS", "20"))

cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True); m["trunk"]["pairformer"]["n_block"] = 48
model = MiniSWAModel(MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"])).to(dev)
model._forced_n_recycle = N_REC
print("MiniSWAModel %.1fM params  L=%d n_atom=%d n_rec=%d" %
      (sum(p.numel() for p in model.parameters())/1e6, L, N_ATOM, N_REC))
batch = _build_precompile_batch(device=dev, msa_depth=16, n_tokens=L, n_atoms=N_ATOM,
                                n_templates=1, num_res_class=int(m["shared"]["num_res_class"]))

def fwd():
    return model.forward(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                         sequence=batch.sequence, structure=batch.structure)

def replay_time(g, iters):
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)

# ---------- INFERENCE graph ----------
model.eval()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for _ in range(3):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fwd()
torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
try:
    g_inf = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.cuda.graph(g_inf):
            _ = fwd()
    med, lo, hi = replay_time(g_inf, ITERS)
    print("INFERENCE (full-model cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("INFERENCE cudagraph capture FAILED:", repr(ex)[:200])

# ---------- TRAINING graph (fwd + distogram loss + bwd) ----------
model.train()
def train_once():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = fwd()
        loss = cal_atom_distogram_loss(logit, batch.structure.atom_pos,
                                       batch.structure.atom_pos_mask,
                                       batch.scheme.atom_to_token_idx_map)
    loss.backward()
    return loss
st2 = torch.cuda.Stream(); st2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st2):
    for _ in range(3):
        model.zero_grad(set_to_none=True); train_once()
torch.cuda.current_stream().wait_stream(st2); torch.cuda.synchronize()
try:
    model.zero_grad(set_to_none=False)  # grads must be allocated & static for capture
    g_tr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_tr):
        train_once()
    med, lo, hi = replay_time(g_tr, ITERS)
    print("TRAINING  (full-model cudagraph): median %.3f ms (min %.3f / max %.3f)" % (med, lo, hi))
except Exception as ex:  # noqa: BLE001
    print("TRAINING cudagraph capture FAILED:", repr(ex)[:200])
print("DONE")
