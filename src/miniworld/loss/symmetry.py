"""Conservative confidence-target correspondence (eager, no gradients).

Match equivalent, identically cropped chains and standard residue name symmetries.
This is not a general ligand automorphism solver: arbitrary CCD graph symmetries
need topology metadata which the current Batch does not retain.
"""
from __future__ import annotations

from itertools import permutations, product
from math import factorial, prod

import numpy as np
import torch

# Coupled aromatic swaps must move both sides together.
_RESIDUE_SWAPS = {
    'ASP': [('OD1', 'OD2')], 'GLU': [('OE1', 'OE2')],
    'ARG': [('NH1', 'NH2')], 'LEU': [('CD1', 'CD2')],
    'VAL': [('CG1', 'CG2')],
    'PHE': [('CD1', 'CD2'), ('CE1', 'CE2')],
    'TYR': [('CD1', 'CD2'), ('CE1', 'CE2')],
}


def _values(value):
    return np.asarray(getattr(value, 'value', value))


@torch.no_grad()
def _align(gt, pred, mask):
    """Proper Kabsch transform; empty and degenerate crops remain finite."""
    w = mask.to(gt.dtype)[:, None]
    count = w.sum().clamp_min(1)
    gc, pc = (gt * w).sum(0) / count, (pred * w).sum(0) / count
    u, _, vh = torch.linalg.svd(((gt - gc) * w).T @ (pred - pc))
    correction = torch.eye(3, device=gt.device, dtype=gt.dtype)
    correction[-1, -1] = torch.det(u @ vh).sign()
    return (gt - gc) @ (u @ correction @ vh) + pc


@torch.no_grad()
def align_confidence_targets(batch, prediction):
    """Return reordered GT coordinates/masks without mutating Batch.

    Exact chain enumeration up to 720 assignments; larger homomers use monotonic
    pair-swap descent. Chains with unequal crop signatures cannot exchange atoms.
    Standard atom-name swaps are evaluated in the globally aligned frame.
    """
    coords = batch.structure.atom_pos.clone()
    masks = batch.structure.atom_pos_mask.bool().clone()
    for b in range(coords.shape[0]):
        exists = batch.structure.atom_mask[b].bool()
        ids = torch.where(exists)[0].tolist()
        names = _values(batch.atom_ids[b]).astype(str)
        components = _values(batch.chem_comp_ids[b]).astype(str)
        a2t = batch.scheme.atom_to_token_idx_map[b].cpu().numpy()
        chain = batch.scheme.atom_to_chain_id[b].cpu().numpy()
        resid = batch.scheme.token_residue_idx[b].cpu().numpy()[a2t]
        entity = batch.scheme.token_entity_id[b].cpu().numpy()[a2t]
        uid = batch.reference.space_uid[b].cpu().numpy()
        bonds = getattr(batch.structure, 'bond_atom_pairs', None)
        bonded = set()
        if bonds is not None:
            for i, j in bonds[b].cpu().tolist():
                if 0 <= i < len(chain) and 0 <= j < len(chain) and i != j:
                    bonded.update((i, j))
        protected_chains = {chain[i] for i in bonded}
        groups = {}
        chain_atoms = {}
        for c in sorted(set(chain[ids].tolist())):
            members = [i for i in ids if chain[i] == c]
            ordered = sorted(members, key=lambda i: (resid[i], components[uid[i]], names[i]))
            signature = tuple((int(resid[i]), components[uid[i]], names[i]) for i in ordered)
            groups.setdefault((int(entity[ordered[0]]), signature, c if c in protected_chains else -1), []).append(c)
            chain_atoms[c] = torch.tensor(ordered, device=coords.device)
        groups = [g for g in groups.values() if len(g) > 1]
        base = torch.arange(coords.shape[1], device=coords.device)
        gt = torch.where(masks[b, :, None], coords[b], 0.).float()
        pred = prediction[b].float()

        def score(p):
            m = masks[b, p]
            aligned = _align(gt[p], pred, m)
            return ((aligned - pred).square().sum(-1) * m).sum().item()

        best, best_score = base, score(base)
        if prod(factorial(len(g)) for g in groups) <= 720:
            for choices in product(*(permutations(g) for g in groups)):
                candidate = base.clone()
                for group, choice in zip(groups, choices):
                    for dst, src in zip(group, choice):
                        candidate[chain_atoms[dst]] = chain_atoms[src]
                value = score(candidate)
                if value < best_score - 1e-6:
                    best, best_score = candidate, value
        else:
            # Bounded heuristic; never accept a worse assignment.
            for _ in range(4):
                improved = False
                for group in groups:
                    for j, dst in enumerate(group):
                        for src in group[j + 1:]:
                            candidate = best.clone()
                            candidate[chain_atoms[dst]] = best[chain_atoms[src]]
                            candidate[chain_atoms[src]] = best[chain_atoms[dst]]
                            value = score(candidate)
                            if value < best_score - 1e-6:
                                best, best_score, improved = candidate, value, True
                if not improved:
                    break
        coords[b], masks[b] = coords[b, best], masks[b, best]
        aligned = _align(coords[b].float(), pred, masks[b])
        for r in sorted(set(uid[ids].tolist())):
            pairs = _RESIDUE_SWAPS.get(components[r], [])
            members = {names[i]: i for i in ids if uid[i] == r}
            if not pairs or any(a not in members or z not in members for a, z in pairs):
                continue
            left = [members[a] for a, z in pairs] + [members[z] for a, z in pairs]
            right = [members[z] for a, z in pairs] + [members[a] for a, z in pairs]
            if bonded.intersection(left):
                continue
            current = ((aligned[left] - pred[left]).square().sum(-1) * masks[b, left]).sum()
            swapped = ((aligned[right] - pred[left]).square().sum(-1) * masks[b, right]).sum()
            if swapped < current:
                coords[b, left] = coords[b, right].clone()
                masks[b, left] = masks[b, right].clone()
    return coords, masks
