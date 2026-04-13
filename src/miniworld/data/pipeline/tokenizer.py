from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt

from miniworld.data.constants import CANONICAL_CHEMCOMPS

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from miniworld.configs import TokenizerConfig
    from miniworld.configs.data import DynamicTokenizationConfig
    from miniworld.data.mols import CIFMolAttached, FragmentedCCDMol


IntArray: TypeAlias = npt.NDArray[np.integer]


def atom_tokenize(cifmol: CIFMolAttached) -> tuple[np.ndarray, np.ndarray]:
    """Build mapping from atom indices to token indices and from token indices to residue indices."""
    # tokens are an intermediate grouping between atoms and residues
    atom_to_residue_idx_map = cifmol.index_table.atom_to_res
    canonical_residue_mask = np.isin(
        cifmol.residues.chem_comp_id.value,
        np.array(list(CANONICAL_CHEMCOMPS)),
    )
    canonical = canonical_residue_mask.astype(bool)
    atom_res = atom_to_residue_idx_map
    n_res = canonical.shape[0]

    counts = np.bincount(atom_res, minlength=n_res)
    token_counts = np.where(canonical, 1, counts)

    token_starts = np.cumsum(
        np.concatenate(([0], token_counts[:-1])),
    )

    order = np.argsort(atom_res, kind="stable")
    sorted_res = atom_res[order]

    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_res)) + 1]
    group_sizes = np.diff(np.r_[group_starts, len(atom_res)])

    offsets_sorted = np.arange(len(atom_res)) - np.repeat(group_starts, group_sizes)

    offsets = np.empty_like(offsets_sorted)
    offsets[order] = offsets_sorted

    atom_to_token_idx_map = token_starts[atom_res] + np.where(
        canonical[atom_res],
        0,
        offsets,
    )

    token_to_residue_idx_map = np.repeat(np.arange(n_res, dtype=np.int64), token_counts)

    return atom_to_token_idx_map, token_to_residue_idx_map


def residue_tokenize(cifmol: CIFMolAttached) -> tuple[np.ndarray, np.ndarray]:
    """Build mapping. ChemComp = Token."""
    return cifmol.index_table.atom_to_res, np.arange(
        len(cifmol.residues),
        dtype=np.int64,
    )


def dynamic_tokenize(  # noqa: PLR0915
    cifmol: CIFMolAttached,
    focus: np.ndarray,  # focus point (3,)
    fragmented_ccd_mols: Mapping[str, Mapping[int, FragmentedCCDMol]],
    rng: np.random.Generator,
    *,
    config: DynamicTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mapping. Dynamic tokenization with distance-based resolution.

    Resolution is sampled from `config.minimum_resolution_ratio` at the focus
    (d=0) and coarsens exponentially with half-life `config.sigma`:

        resolution(d) = 1 - (1 - min_res) * exp(-ln2 * d / sigma)

    resolution=0 → atomize (one token per atom)
    resolution=1 → residue-level (one token per residue)
    Values in between map uniformly to the available pre-computed merge levels
    in `fragmented_ccd_mols`.

    Args:
        cifmol: Input structure.
        focus: 3-D focus point (Å).
        fragmented_ccd_mols: {ccd_name: {merge_level: FragmentedCCDMol}}.
        rng: Random number generator.
        config: Tokenization hyper-parameters (sigma, resolution distribution).

    Returns:
        atom_to_token_idx_map: shape (n_atoms,) — token index per atom.
        token_to_residue_idx_map: shape (n_tokens,) — residue index per token.

    """
    atom_to_res = cifmol.index_table.atom_to_res
    n_residues = len(cifmol.residues)
    xyz = cifmol.atoms.xyz.value  # (n_atoms, 3)

    # --- Sample min_resolution from the configured distribution ---
    probs = np.asarray(config.minimum_resolution_ratio, dtype=float)
    probs /= probs.sum()
    choice = int(rng.choice(3, p=probs))
    if choice == 0:
        min_res = 0.0  # atomize at focus
    elif choice == 2:
        min_res = 1.0  # residue-level everywhere
    else:
        min_res = float(rng.uniform(0.0, 1.0))

    # --- Per-atom distances; invalid coords (NaN/inf) → inf ---
    valid = np.isfinite(xyz).all(axis=1)
    atom_dists = np.where(valid, np.linalg.norm(xyz - focus, axis=1), np.inf)

    # --- Per-residue: minimum distance across atoms (invalid residue → inf) ---
    res_dists = np.full(n_residues, np.inf)
    np.minimum.at(res_dists, atom_to_res, atom_dists)

    # --- Sample sigma: flat (inf) with prob sigma_flat_prob, else LogUniform ---
    if rng.random() < config.sigma_flat_prob:
        sigma = np.inf  # resolution = min_res everywhere
    else:
        log_sigma = rng.uniform(np.log(config.sigma_min), np.log(config.sigma_max))
        sigma = np.exp(log_sigma)
    LOGGER.debug("min_res=%.3f sigma=%.3f focus=%s", min_res, sigma, focus)

    # --- Exponential schedule: half-life = sigma ---
    # sigma=inf → flat at min_res everywhere (avoids inf/inf = NaN)
    if np.isinf(sigma):
        resolutions = np.full(n_residues, min_res)
    else:
        resolutions = 1.0 - (1.0 - min_res) * np.exp(-np.log(2.0) * res_dists / sigma)
    # Residues with no valid atoms (res_dists=inf) → coarsest resolution
    resolutions = np.where(np.isfinite(res_dists), resolutions, 1.0)

    # --- Build token assignments ---
    ccd_names = cifmol.residues.chem_comp_id.value
    n_atoms = len(cifmol.atoms)
    atom_to_token = np.empty(n_atoms, dtype=np.int64)
    token_to_res: list[int] = []
    token_offset = 0

    for r in range(n_residues):
        atom_indices = np.where(atom_to_res == r)[0]
        ccd_name = str(ccd_names[r])

        if ccd_name not in fragmented_ccd_mols:
            # Unknown CCD : one token per residue
            atom_to_token[atom_indices] = token_offset
            token_to_res.append(r)
            token_offset += 1
            continue

        frag_dict = fragmented_ccd_mols[ccd_name]
        available = sorted(frag_dict.keys())

        # Map resolution [0, 1] uniformly onto the available merge indices
        idx = max(
            0,
            min(round(float(resolutions[r]) * (len(available) - 1)), len(available) - 1),
        )
        merge_val = available[idx]

        frag_mol = frag_dict[merge_val]
        local_frag = (
            frag_mol.index_table.atom_to_res
        )  # atom_idx → frag_idx (within residue)
        n_frags = len(frag_mol.residues)

        if len(local_frag) != len(atom_indices):
            # Atom-count mismatch between CIF and CCD template: fall back
            atom_to_token[atom_indices] = token_offset
            token_to_res.append(r)
            token_offset += 1
            continue

        atom_to_token[atom_indices] = token_offset + local_frag
        token_to_res.extend([r] * n_frags)
        token_offset += n_frags

    return atom_to_token, np.array(token_to_res, dtype=np.int64)


class Tokenizer:
    """Tokenizer for converting chemcomp into model input features."""

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize the tokenizer with the specified tokenization level."""
        self.level = config.level
        self.dynamic_config = config.dynamic_config
        self.rng = np.random.default_rng(config.seed)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic tokenization."""
        self.rng = np.random.default_rng(epoch)

    def tokenize(
        self,
        cifmol: CIFMolAttached,
        **kwargs: Any,
    ) -> tuple[IntArray, IntArray]:
        """Tokenize the given CIFMolAttached into (atom_to_token_idx_map, token_residue_idx)."""
        if self.level == "atom":
            return atom_tokenize(cifmol)
        if self.level == "residue":
            return residue_tokenize(cifmol)
        if self.level == "dynamic":
            dynamic_config = kwargs.pop("config", None)
            if dynamic_config is None:
                dynamic_config = self.dynamic_config
            if dynamic_config is None:
                msg = "Dynamic tokenization requires TokenizerConfig.dynamic_config."
                raise ValueError(msg)
            return dynamic_tokenize(
                cifmol,
                rng=self.rng,
                config=dynamic_config,
                **kwargs,
            )
        if self.level == "lte":
            msg = "LTE tokenization is not implemented yet."
            raise NotImplementedError(msg)
        msg = f"Invalid tokenization level: {self.level}"
        raise ValueError(msg)
