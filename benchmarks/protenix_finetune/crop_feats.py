"""Crop a saved Protenix feature dict to its first C tokens (and their atoms), like a training crop of crop_size C.
Token axes (size N_token) and atom axes (size N_atom) are sliced; atoms must be ordered by token so the kept atoms are a prefix."""
import os, sys, torch
from pathlib import Path
FEATS = Path(os.environ.get("PFX_FEATS", Path(__file__).resolve().parent / "feats"))
src, C = sys.argv[1], int(sys.argv[2])
d = torch.load(FEATS / f"{src}.pt", weights_only=False)
f = d["input_feature_dict"]
NT, NA = int(d["N_token"]), int(d["N_atom"])
a2t = f["atom_to_token_idx"]
assert bool((a2t[1:] >= a2t[:-1]).all()), "atoms not ordered by token"
CA = int((a2t < C).sum())
def crop(v):
    if isinstance(v, dict):
        return {k: crop(x) for k, x in v.items()}
    if not torch.is_tensor(v) or v.dim() == 0:
        return v
    for ax, n in enumerate(v.shape):
        if n == NT and NT != NA:
            v = v.narrow(ax, 0, C)
        elif n == NA:
            v = v.narrow(ax, 0, CA)
    return v.contiguous()
g = {k: crop(v) for k, v in f.items()}
assert int(g["frame_atom_index"].max()) < CA
out = dict(d); out["input_feature_dict"] = g
out["N_token"] = torch.tensor([C]); out["N_atom"] = torch.tensor([CA]); out["sample_name"] = f"{src}_c{C}"
torch.save(out, FEATS / f"{src}_c{C}.pt")
print(f"{src}: {NT} tokens / {NA} atoms -> {C} tokens / {CA} atoms, N_msa {int(d['N_msa'])}; chains kept {torch.unique(g['asym_id'], return_counts=True)}")
