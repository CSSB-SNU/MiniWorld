from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from miniworld.data.constants import CANONICAL_CHEMCOMPS

if TYPE_CHECKING:
    from miniworld.configs import TokenizerConfig
    from miniworld.data.mols import CIFMolAttached


IntArray: TypeAlias = npt.NDArray[np.integer]


class TokenizeFn(Protocol):
    """Protocol for tokenization functions."""

    def __call__(self, cifmol: CIFMolAttached) -> tuple[IntArray, IntArray]:
        """Convert cifmol into (atom_to_token_idx_map, token_residue_idx)."""
        ...


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


class Tokenizer:
    """Tokenizer for converting chemcomp into model input features."""

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize the tokenizer with the specified tokenization level."""
        self.level = config.level
        self.genereor = np.random.default_rng(config.seed)
        self._tokenize_fn: TokenizeFn = self._resolve_tokenize_fn(self.level)

    def tokenize(self, cifmol: CIFMolAttached) -> tuple[IntArray, IntArray]:
        """Tokenize the given CIFMolAttached into (atom_to_token_idx_map, token_residue_idx)."""
        return self._tokenize_fn(cifmol)

    @staticmethod
    def _resolve_tokenize_fn(
        level: Literal["atom", "dynamic", "lte", "residue"],
    ) -> TokenizeFn:
        if level == "atom":
            return atom_tokenize
        if level == "residue":
            return residue_tokenize
        if level == "dynamic":
            msg = "Dynamic tokenization is not implemented yet."
            raise NotImplementedError(msg)
        if level == "lte":
            msg = "LTE tokenization is not implemented yet."
            raise NotImplementedError(msg)
        msg = f"Invalid tokenization level: {level}"
        raise ValueError(msg)
