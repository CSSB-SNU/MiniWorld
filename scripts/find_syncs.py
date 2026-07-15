"""Pinpoint every device->host sync in the MiniSWAModel forward — these are what
invalidate a CUDA-graph capture. Warm up once (JIT/autotune settle), then run under
torch.cuda.set_sync_debug_mode('error') so the FIRST sync raises with a traceback at
the exact line. Fix -> repeat. Runs eval (inference) then a train step."""
from __future__ import annotations
import sys, traceback
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
cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True); m["trunk"]["pairformer"]["n_block"] = 48
model = MiniSWAModel(MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"])).to(dev)
model._forced_n_recycle = 2
batch = _build_precompile_batch(device=dev, msa_depth=16, n_tokens=256, n_atoms=1024,
                                n_templates=1, num_res_class=int(m["shared"]["num_res_class"]))

def fwd():
    return model.forward(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                         sequence=batch.sequence, structure=batch.structure)

# ---- warm up (compile/autotune/JIT settle; also warms the packing cache) ----
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    fwd(); fwd()
torch.cuda.synchronize()
print("warmup done")

# ---- INFERENCE: catch the first sync ----
torch.cuda.set_sync_debug_mode("error")
try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fwd()
    torch.cuda.synchronize()
    print("INFERENCE: NO device->host sync detected")
except Exception:  # noqa: BLE001
    torch.cuda.set_sync_debug_mode("default")
    print("INFERENCE first sync:\n" + "".join(traceback.format_exc()))
torch.cuda.set_sync_debug_mode("default")

# ---- TRAINING: catch the first sync (fwd+loss+bwd) ----
model.train()
model.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):  # warm train path once
    lg = fwd()
    cal_atom_distogram_loss(lg, batch.structure.atom_pos, batch.structure.atom_pos_mask,
                            batch.scheme.atom_to_token_idx_map).backward()
torch.cuda.synchronize()
model.zero_grad(set_to_none=True)
torch.cuda.set_sync_debug_mode("error")
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = fwd()
        loss = cal_atom_distogram_loss(lg, batch.structure.atom_pos,
                                       batch.structure.atom_pos_mask,
                                       batch.scheme.atom_to_token_idx_map)
    loss.backward()
    torch.cuda.synchronize()
    print("TRAINING: NO device->host sync detected")
except Exception:  # noqa: BLE001
    torch.cuda.set_sync_debug_mode("default")
    print("TRAINING first sync:\n" + "".join(traceback.format_exc()))
torch.cuda.set_sync_debug_mode("default")
print("DONE")
