"""Enumerate every LayerNorm-like module in the built MiniSWAModel and classify:
native torch.nn.LayerNorm vs team_gm primitives.LayerNorm, and the latter's
`implementation` (PYTORCH=native fp32 vs TRITON=ops.layer_norm our kernel)."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path("scripts").resolve()))
from miniworld.configs.models import AtomSWAConfig
from miniworld.models.distogram_only import MiniSWAModel
from team_gm.modules.primitives import LayerNorm as PrimLayerNorm

cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True)
m["trunk"]["pairformer"]["n_block"] = 48
swa_cfg = MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"],
)
model = MiniSWAModel(swa_cfg)

native = Counter()
prim_pytorch = Counter()
prim_triton = Counter()
for name, mod in model.named_modules():
    cls = type(mod)
    is_prim = isinstance(mod, PrimLayerNorm)
    is_native = isinstance(mod, torch.nn.LayerNorm) and not is_prim
    if is_native:
        # group by the parent path up to the block index removed
        key = name.split(".")[-2] if "." in name else name
        native[key] += 1
    elif is_prim:
        impl = str(getattr(mod, "implementation", "?"))
        top = name.split(".")[2] if len(name.split(".")) > 2 else name
        (prim_triton if "TRITON" in impl or "CUEQ" in impl else prim_pytorch)[f"{top}:{impl}"] += 1

print("=== raw torch.nn.LayerNorm (NATIVE fp32 under autocast) ===")
for k, v in native.most_common():
    print(f"  {v:4d}  ...{k}")
print("  TOTAL native nn.LayerNorm modules:", sum(native.values()))
print("=== primitives.LayerNorm impl=PYTORCH (also native) ===")
for k, v in prim_pytorch.most_common():
    print(f"  {v:4d}  {k}")
print("=== primitives.LayerNorm impl=TRITON/CUEQ (our ops.layer_norm) ===")
for k, v in prim_triton.most_common():
    print(f"  {v:4d}  {k}")
