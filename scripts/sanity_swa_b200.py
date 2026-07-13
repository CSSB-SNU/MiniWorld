"""B200 sanity for MiniSWAModel: build config (3 SWA embedder / 4 MSA / 48 Pairformer),
run forward + distogram loss + one backward (train step). Reuses _build_precompile_batch
for dummy features and cal_atom_distogram_loss exactly as Client.loss_fn does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path("scripts").resolve()))

from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa: F401  (ensures registration)
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0)
dev = torch.device("cuda")

# --- load the medium_distogram model subtree, then adapt to MiniSWAModel.Config ---
cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True)

# input_feat_embbeder n_block=3 and msa_module n_block=4 already in yaml; bump pairformer 16->48
m["trunk"]["pairformer"]["n_block"] = 48
print("blocks: input_feat=%d  msa=%d  pairformer=%d"
      % (m["input_feat_embbeder"]["n_block"],
         m["trunk"]["msa_module"]["n_block"],
         m["trunk"]["pairformer"]["n_block"]))

swa_cfg = MiniSWAModel.Config(
    shared=m["shared"],
    input_feat_embbeder=m["input_feat_embbeder"],
    # ESMFold2-style atom front-end: FA4 backend, large window == global attention
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"],
)

model = MiniSWAModel(swa_cfg).to(dev)
model.train()
n_params = sum(p.numel() for p in model.parameters())
print("MiniSWAModel built: %.1fM params" % (n_params / 1e6))

# --- dummy batch (same builder the train warmup uses) ---
batch = _build_precompile_batch(
    device=dev,
    msa_depth=16,
    n_tokens=64,
    n_atoms=256,
    n_templates=1,
    num_res_class=int(m["shared"]["num_res_class"]),
)

# --- forward (replicates Client.loss_fn under fabric's bf16-mixed autocast) ---
model._forced_n_recycle = 2  # noqa: SLF001  (deterministic; exercises recycle loop)
with torch.autocast("cuda", dtype=torch.bfloat16):
    logit = model.forward(
        msa=batch.msa,
        reference=batch.reference,
        scheme=batch.scheme,
        sequence=batch.sequence,
        structure=batch.structure,
    )
    loss = cal_atom_distogram_loss(
        logit,
        batch.structure.atom_pos,
        batch.structure.atom_pos_mask,
        batch.scheme.atom_to_token_idx_map,
    )
print("forward OK  logit:", tuple(logit.shape), logit.dtype,
      "finite=%s" % bool(torch.isfinite(logit).all()))
print("distogram_loss:", float(loss))

# --- one train step: backward + grad-flow check ---
loss.backward()
g_tot = g_none = 0
for p in model.parameters():
    if p.requires_grad:
        g_tot += 1
        if p.grad is None:
            g_none += 1
gn = torch.sqrt(sum((p.grad.float() ** 2).sum()
                    for p in model.parameters() if p.grad is not None))
print("backward OK  params_with_grad=%d/%d  grad_norm=%.4f" % (g_tot - g_none, g_tot, float(gn)))
assert g_none == 0, "%d params got no grad" % g_none
assert torch.isfinite(gn), "grad norm non-finite"
print("B200 SANITY PASS: MiniSWAModel fwd + distogram loss + 1 train-step OK")
