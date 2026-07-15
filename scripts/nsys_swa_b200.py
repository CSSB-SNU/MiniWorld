"""nsys component breakdown for MiniSWAModel. NVTX ranges (via forward hooks) wrap
the major stages — input_feature_embedder (SWA/FA4), msa_module, pairformer_blocks,
distogram_head — so `nsys stats --report nvtx_gpu_proj_sum` projects GPU kernel time
onto each. Run EAGER (no compile/cudagraph) so kernels attribute cleanly to module
boundaries. Range names are prefixed by phase (infer/train). The trunk runs n_recycle
times, so a component's projected time is the SUM across recycles per iteration.
"""
from __future__ import annotations

import os
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
L = int(os.environ.get("L_TOK", "256"))
N_ATOM = int(os.environ.get("N_ATOM", str(L * 4)))
N_REC = int(os.environ.get("N_REC", "4"))
ITERS = int(os.environ.get("ITERS", "3"))

cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True)
m["trunk"]["pairformer"]["n_block"] = 48
swa_cfg = MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"],
)
model = MiniSWAModel(swa_cfg).to(dev)
model._forced_n_recycle = N_REC  # noqa: SLF001

batch = _build_precompile_batch(
    device=dev, msa_depth=16, n_tokens=L, n_atoms=N_ATOM, n_templates=1,
    num_res_class=int(m["shared"]["num_res_class"]),
)

PHASE = "infer"
_COMPONENTS = {
    "input_feature_embedder": model.input_feature_embedder,
    "msa_module": model.msa_module,
    "pairformer": model.pairformer_blocks,
    "distogram_head": model.distogram_head,
}
for cname, mod in _COMPONENTS.items():
    def _pre(m_, inp, _c=cname):
        torch.cuda.nvtx.range_push(f"{PHASE}/{_c}")
    def _post(m_, inp, out, _c=cname):
        torch.cuda.nvtx.range_pop()
    mod.register_forward_pre_hook(_pre)
    mod.register_forward_hook(_post)


def fwd():
    return model.forward(
        msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
        sequence=batch.sequence, structure=batch.structure,
    )


# warmup (kernels compiled/autotuned before capture)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    fwd()
model.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    lg = fwd()
    cal_atom_distogram_loss(lg, batch.structure.atom_pos, batch.structure.atom_pos_mask,
                            batch.scheme.atom_to_token_idx_map).backward()
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# ---- profiled region ----
PHASE = "infer"
model.eval()
torch.cuda.nvtx.range_push("INFER")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(ITERS):
        fwd()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

PHASE = "train"
model.train()
torch.cuda.nvtx.range_push("TRAIN")
for _ in range(ITERS):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = fwd()
        loss = cal_atom_distogram_loss(lg, batch.structure.atom_pos,
                                       batch.structure.atom_pos_mask,
                                       batch.scheme.atom_to_token_idx_map)
    loss.backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("NSYS RUN DONE  L=%d n_rec=%d iters=%d" % (L, N_REC, ITERS))
