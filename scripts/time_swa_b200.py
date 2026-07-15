"""B200 latency for MiniSWAModel — inference and training measured SEPARATELY,
eager vs torch.compile, CUDA-event timed with warmup. Shipped autotune caches are
used (MINIWORLD_RUN_AUTOTUNE unset), so kernels pick tuned configs, not full grid.
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path("scripts").resolve()))

from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa: F401
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0)
dev = torch.device("cuda")

L = int(os.environ.get("L_TOK", "256"))        # crop size (n_tokens)
N_ATOM = int(os.environ.get("N_ATOM", str(L * 4)))
N_REC = int(os.environ.get("N_REC", "4"))       # fixed recycles for reproducible timing
COMPILE = os.environ.get("COMPILE", "1") == "1"
ITERS = int(os.environ.get("ITERS", "10"))
WARMUP = int(os.environ.get("WARMUP", "5"))

cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True)
m["trunk"]["pairformer"]["n_block"] = 48
swa_cfg = MiniSWAModel.Config(
    shared=m["shared"],
    input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"],
)
model = MiniSWAModel(swa_cfg).to(dev)
model._forced_n_recycle = N_REC  # noqa: SLF001
n_params = sum(p.numel() for p in model.parameters())
print("MiniSWAModel: %.1fM params  L=%d n_atom=%d n_recycle=%d compile=%s"
      % (n_params / 1e6, L, N_ATOM, N_REC, COMPILE))

batch = _build_precompile_batch(
    device=dev, msa_depth=16, n_tokens=L, n_atoms=N_ATOM, n_templates=1,
    num_res_class=int(m["shared"]["num_res_class"]),
)


def fwd():
    return model.forward(
        msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
        sequence=batch.sequence, structure=batch.structure,
    )


MODE = os.environ.get("CMODE", "reduce-overhead")  # reduce-overhead=cudagraph, or "default"
run = fwd
if COMPILE:
    run = torch.compile(fwd, mode=MODE) if MODE != "default" else torch.compile(fwd)
_CUDAGRAPH = COMPILE and MODE == "reduce-overhead"


def _mark():
    if _CUDAGRAPH:
        torch.compiler.cudagraph_mark_step_begin()


def time_loop(step, warmup, iters):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)


# ---- INFERENCE (eval, no grad) ----
model.eval()
def infer_step():
    _mark()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        run()
med, lo, hi = time_loop(infer_step, WARMUP, ITERS)
print("INFERENCE : median %.3f ms  (min %.3f / max %.3f)" % (med, lo, hi))

# ---- TRAINING (fwd + distogram loss + bwd) ----
model.train()
def train_step():
    _mark()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = run()
        if _CUDAGRAPH:
            logit = logit.clone()  # detach from cudagraph pool before loss/bwd read it
        loss = cal_atom_distogram_loss(
            logit, batch.structure.atom_pos, batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
        )
    loss.backward()
med, lo, hi = time_loop(train_step, WARMUP, ITERS)
print("TRAINING  : median %.3f ms  (min %.3f / max %.3f)" % (med, lo, hi))
print("DONE")
