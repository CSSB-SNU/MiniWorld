"""Optional ligand-geometry steering during diffusion sampling.

Small molecules (non-polymer / branched chains) come out of the diffusion
sampler with the right *pose* but frequently distorted *internal* geometry —
stretched bonds, broken rings — because the network has comparatively little
per-atom supervision for them (especially when the ligand is a single
residue-level token). Their topology, however, is known exactly from the CCD.

This module captures that known geometry as a set of harmonic **distance
restraints** and projects each diffusion ``x0_hat`` estimate back toward it:

  * **1-2 (bonds)** and **1-3 (angles, encoded as the 1-3 distance)** pairs are
    restrained to their CCD ideal lengths. These are the rigid DOF.
  * **1-4 and beyond (torsions) are left free** — we never restrain them, so the
    ligand's rotatable bonds and its rigid-body pose stay whatever the model
    predicted.
  * a weak **tether** to the model's prediction keeps the relaxed ligand near
    the predicted position/orientation (distance restraints alone are
    rotation/translation/reflection invariant, so the tether also pins the
    correct enantiomer the model already chose).

Because every term is a function of interatomic distances, the potential is
SE(3)-invariant: it fixes internal shape without biasing the global placement
or the ligand's binding pose.

The restraint set is static for a batch (built once, reused every step). The
injection point is :func:`miniworld.inference.solver.sample_trajectory`, which
applies it to ``x_pred`` just before the renoise — see that function. Gated by
``infer.ligand_potential`` in the run config; off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from miniworld.data.io.load import load_raw_data
from miniworld.data.mols import CCDMol

# Entity-type indices that carry a small-molecule topology worth steering
# (miniworld.data.constants.mapping: L=LIGAND, B=BRANCHED).
_LIGAND_ENTITY_IDS = (6, 7)


@dataclass
class LigandRestraint:
    """Static harmonic distance restraints for every ligand atom in a batch.

    All index tensors are **global atom indices** into the ``L_atom`` axis of
    the sample (the same axis as ``x_pred``). ``pair_i/pair_j`` enumerate the
    restrained 1-2 and 1-3 pairs; ``pair_d`` is the CCD ideal distance and
    ``pair_w`` the per-pair weight (bond vs angle). ``atom_index`` is the set
    of all ligand atoms (the only atoms the projection is allowed to move).
    """

    atom_index: torch.Tensor  # (n_lig,) long
    pair_i: torch.Tensor      # (P,) long
    pair_j: torch.Tensor      # (P,) long
    pair_d: torch.Tensor      # (P,) float — CCD ideal distance
    pair_w: torch.Tensor      # (P,) float — restraint weight

    def to(self, device: torch.device) -> "LigandRestraint":
        return LigandRestraint(
            atom_index=self.atom_index.to(device),
            pair_i=self.pair_i.to(device),
            pair_j=self.pair_j.to(device),
            pair_d=self.pair_d.to(device),
            pair_w=self.pair_w.to(device),
        )

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.numel())


def _clean_model_xyz(m: CCDMol) -> np.ndarray:
    raw = np.asarray(m.atoms.model_xyz.value, dtype=object)
    miss = (raw == "?") | (raw == ".")
    raw[miss] = 0.0
    xyz = raw.astype(np.float32)
    return np.nan_to_num(xyz, nan=0.0)


def _residue_pairs(
    xyz: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    w_bond: float,
    w_angle: float,
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Build (local) 1-2 and 1-3 distance restraints for one CCD residue.

    Returns ``(pairs, dists, weights)`` with 0-based atom indices local to the
    residue. 1-3 pairs are derived from the bond adjacency (two bonds sharing a
    central atom); torsion (1-4+) pairs are intentionally never emitted.
    """
    n = xyz.shape[0]
    pairs: list[tuple[int, int]] = []
    dists: list[float] = []
    weights: list[float] = []

    # 1-2 bonds
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src.tolist(), dst.tolist()):
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        pairs.append((lo, hi))
        dists.append(float(np.linalg.norm(xyz[lo] - xyz[hi])))
        weights.append(w_bond)
        adj[a].append(b)
        adj[b].append(a)

    # 1-3 (angles) — neighbours of a shared central atom
    seen: set[tuple[int, int]] = set(pairs)
    for center in range(n):
        nbrs = adj[center]
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                a, b = nbrs[ii], nbrs[jj]
                if a == b:
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                if (lo, hi) in seen:
                    continue
                seen.add((lo, hi))
                pairs.append((lo, hi))
                dists.append(float(np.linalg.norm(xyz[lo] - xyz[hi])))
                weights.append(w_angle)
    return pairs, dists, weights


def build_ligand_restraints(
    batch,
    chain_residue_ccds: dict[int, list[str]],
    ccd_db_path: str | Path,
    *,
    w_bond: float = 1.0,
    w_angle: float = 0.5,
) -> LigandRestraint | None:
    """Assemble a :class:`LigandRestraint` for every ligand chain in ``batch``.

    ``chain_residue_ccds`` maps a query chain **index** to the ordered list of
    CCD codes making up that chain (one entry per residue). Only chains whose
    ``batch.chain.entity_type`` is LIGAND/BRANCHED are restrained; the caller
    may pass extra chains and they are ignored. Returns ``None`` when there is
    no ligand to steer.

    Bond connectivity and ideal coordinates come from the CCD LMDB; atom order
    within a residue matches the dataloader (CCDLookup preserves CCD order), so
    each residue's atoms are the contiguous block following the previous one.
    """
    ccd_db_path = Path(ccd_db_path)
    atom_to_chain = batch.scheme.atom_to_chain_id[0].cpu().numpy()
    entity_type = batch.chain.entity_type[0].cpu().numpy()

    all_lig_atoms: list[int] = []
    pi: list[int] = []
    pj: list[int] = []
    pd: list[float] = []
    pw: list[float] = []

    bond_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for chain_idx in sorted(chain_residue_ccds):
        if chain_idx >= entity_type.shape[0]:
            continue
        if int(entity_type[chain_idx]) not in _LIGAND_ENTITY_IDS:
            continue
        chain_atoms = np.nonzero(atom_to_chain == chain_idx)[0]
        if chain_atoms.size == 0:
            continue
        chain_atoms.sort()
        cursor = int(chain_atoms[0])  # global index of this chain's first atom

        for ccd in chain_residue_ccds[chain_idx]:
            if ccd not in bond_cache:
                raw = load_raw_data(ccd, ccd_db_path)
                if raw is None:
                    msg = f"CCD entry {ccd!r} not found in {ccd_db_path}."
                    raise KeyError(msg)
                m = CCDMol.from_bytes(raw)
                xyz = _clean_model_xyz(m)
                src = np.asarray(m.atoms.bond_type.src)
                dst = np.asarray(m.atoms.bond_type.dst)
                bond_cache[ccd] = (xyz, src, dst)
            xyz, src, dst = bond_cache[ccd]
            n = xyz.shape[0]

            pairs, dists, weights = _residue_pairs(xyz, src, dst, w_bond, w_angle)
            for (a, b), d, w in zip(pairs, dists, weights):
                pi.append(cursor + a)
                pj.append(cursor + b)
                pd.append(d)
                pw.append(w)
            all_lig_atoms.extend(range(cursor, cursor + n))
            cursor += n

    if not pi:
        return None

    return LigandRestraint(
        atom_index=torch.tensor(sorted(set(all_lig_atoms)), dtype=torch.long),
        pair_i=torch.tensor(pi, dtype=torch.long),
        pair_j=torch.tensor(pj, dtype=torch.long),
        pair_d=torch.tensor(pd, dtype=torch.float32),
        pair_w=torch.tensor(pw, dtype=torch.float32),
    )


def apply_ligand_restraint(
    x_pred: torch.Tensor,
    restraint: LigandRestraint,
    *,
    n_steps: int = 20,
    lr: float = 0.05,
    w_tether: float = 0.1,
) -> torch.Tensor:
    """Project ``x_pred`` toward valid ligand geometry by harmonic GD.

    Operates only on ligand atoms (``restraint.atom_index``); every other atom
    is untouched. Runs ``n_steps`` of analytic gradient descent on

        U = Σ_pairs w·(‖x_i − x_j‖ − d0)²  +  w_tether·‖x_lig − x_pred_lig‖²

    using manual gradients so it is safe under ``torch.inference_mode``. The
    tether target is the *incoming* prediction, so the ligand stays at the
    predicted pose while its internal distances snap to CCD ideal.

    ``x_pred`` is ``(A, L_atom, 3)``; returns a new tensor of the same shape.
    """
    r = restraint.to(x_pred.device)
    x = x_pred.clone()
    lig = r.atom_index
    tether_target = x_pred[:, lig, :].clone()
    pd = r.pair_d.to(x.dtype)
    pw = r.pair_w.to(x.dtype)

    for _ in range(n_steps):
        grad = torch.zeros_like(x)
        diff = x[:, r.pair_i, :] - x[:, r.pair_j, :]          # (A, P, 3)
        dist = diff.norm(dim=-1).clamp_min(1e-6)              # (A, P)
        coeff = (2.0 * pw * (dist - pd) / dist).unsqueeze(-1)  # (A, P, 1)
        g = coeff * diff
        grad.index_add_(1, r.pair_i, g)
        grad.index_add_(1, r.pair_j, -g)
        grad[:, lig, :] = grad[:, lig, :] + 2.0 * w_tether * (x[:, lig, :] - tether_target)
        x[:, lig, :] = x[:, lig, :] - lr * grad[:, lig, :]

    return x
