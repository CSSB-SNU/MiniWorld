"""Capture the B200 (sm100) triton autotune cache for the SPLIT transition kernels
(transition_split_fwd / transition_split_bwd). The module capture builder benches
the FUSED b2b transition impl, which never fires the split kernels the deployed
trunk uses on B200. Here we drive TritonTransitionFunction (the split path) directly
with the model's transition expansion n, across the production d_pair range and two
representative M buckets (small crop like the sanity + large crop like training) so
the shipped cache covers the (GROUP_M, n, N) buckets the model actually hits.

No model forward is run (that would trigger full-grid autotune for all 48+4 trunk
blocks under MINIWORLD_RUN_AUTOTUNE=1 and take far too long) — the model is only
built to read the transition expansion n off its weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path("scripts").resolve()))

from miniworld_kernels.autotune import capture
from miniworld_kernels.kernels.transition.triton.main import TritonTransitionFunction

from miniworld.configs.models import AtomSWAConfig
from miniworld.models.distogram_only import MiniSWAModel

torch.manual_seed(0)
dev = torch.device("cuda")

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

# discover transition expansion n from a trunk transition weight ([n*d, d]) at d_pair
d_pair = int(m["shared"]["d_pair"])
n_expand = None
for name, p in model.named_parameters():
    if "transition" in name.lower() and p.dim() == 2 and p.shape[1] == d_pair \
            and p.shape[0] % d_pair == 0 and p.shape[0] // d_pair in (2, 4, 8):
        n_expand = p.shape[0] // d_pair
        print("discovered transition n=%d from %s %s" % (n_expand, name, tuple(p.shape)))
        break
if n_expand is None:
    n_expand = 4
    print("fallback transition n=4")
del model
torch.cuda.empty_cache()

capture.install()
capture.reset()

# split fwd+bwd across production d_pair range x two M buckets (small crop / train crop)
import os
_NS = [int(x) for x in os.environ.get("CAP_NS", "128").split(",")]
_LS = [int(x) for x in os.environ.get("CAP_LS", "64").split(",")]
for N in _NS:
    for L in _LS:        # M = L*L flattened token-pair rows; bucket_of coarsens M
        M = L * L
        x = torch.randn(M, N, device=dev, dtype=torch.bfloat16, requires_grad=True)
        ea = torch.randn(n_expand * N, N, device=dev, dtype=torch.bfloat16, requires_grad=True)
        eb = torch.randn(n_expand * N, N, device=dev, dtype=torch.bfloat16, requires_grad=True)
        sq = torch.randn(N, n_expand * N, device=dev, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = TritonTransitionFunction.apply(x, ea, eb, sq, n_expand)
        out.sum().backward()
        print("split fwd+bwd done  d_pair=%d L=%d M=%d n=%d" % (N, L, M, n_expand))

print("=== capture summary ===")
print(capture.summary())
written = capture.flush(top_k=5)
print("=== flushed ===")
for op, dtype, bucket, ncfg, fp in written:
    print("  %s [%s|%s] %d cfgs -> %s" % (op, dtype, bucket, ncfg, fp))
