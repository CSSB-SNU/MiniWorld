"""Bond-geometry metrics for predicted coordinates (x0-hat or a final structure), no ground truth needed.

Ideal covalent bonds = pairs of atoms of the SAME reference residue (``reference.space_uid``) whose distance in the CCD reference conformer
(``reference.pos``) is below 1.9 A (1-2 neighbours are 1.2-1.8 A, 1-3 neighbours >= 2.2 A); the reference distance is the ideal length.
Inter-residue (peptide) bonds are not included. Clashes = atom pairs of non-adjacent residues closer than 2.0 A in the prediction.
"""
from __future__ import annotations

import torch

BOND_MAX = 1.9
BOND_MIN = 0.5
CLASH_DIST = 2.0


@torch.no_grad()
def ideal_bonds(ref_pos: torch.Tensor, space_uid: torch.Tensor, valid: torch.Tensor):
    """ref_pos [L,3], space_uid [L] (int), valid [L] (bool) -> (i, j, ideal_len) of the covalent bonds."""
    pos = ref_pos.float()
    n = pos.shape[0]
    d = torch.cdist(pos, pos)
    same = space_uid[:, None] == space_uid[None, :]
    ok = valid[:, None] & valid[None, :]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pos.device), diagonal=1)
    sel = same & ok & upper & (d < BOND_MAX) & (d > BOND_MIN)
    i, j = sel.nonzero(as_tuple=True)
    return i, j, d[i, j]


@torch.no_grad()
def bond_metrics(x: torch.Tensor, i, j, ideal, space_uid=None, valid=None, with_clash=True) -> dict:
    """x [L,3] -> dict of bond deviation statistics (Angstrom) and the clash count."""
    x = x.float()
    dev = ((x[i] - x[j]).norm(dim=-1) - ideal).abs()
    out = {
        "n_bonds": int(dev.numel()),
        "mean_abs_dev": float(dev.mean()) if dev.numel() else float("nan"),
        "frac_gt_0.1": float((dev > 0.1).float().mean()) if dev.numel() else float("nan"),
        "frac_gt_0.3": float((dev > 0.3).float().mean()) if dev.numel() else float("nan"),
        "max_dev": float(dev.max()) if dev.numel() else float("nan"),
    }
    if with_clash and space_uid is not None:
        d = torch.cdist(x, x)
        far = (space_uid[:, None] - space_uid[None, :]).abs() > 1
        ok = valid[:, None] & valid[None, :]
        upper = torch.triu(torch.ones_like(far), diagonal=1)
        out["n_clash"] = int(((d < CLASH_DIST) & far & ok & upper).sum())
    return out
