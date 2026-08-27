"""Inference-only crop-focus override that biases toward modified residues.

The two training-time dataloaders (``dataloader.BioMolData`` and
``dataloader_explicit.BioMolData``) are not touched. Instead this module
exposes a single factory ``apply_modified_focus`` that swaps the focus
selection on an existing instance. Both parents share the same
``Tokenizer`` (``miniworld.data.pipeline.Tokenizer``) and crop-pipeline
signature, so one override body works for both.

Usage in an inference script:

    from miniworld.data.dataloader.dataloader_infer import apply_modified_focus
    infer_dataset = BioMolData(config)
    apply_modified_focus(infer_dataset)
    # iterate the dataset / dataloader as usual.

On a structure with no modified residue in any priority bucket,
``select_modified_focus`` raises ``NoModifiedResidueError`` — a sibling of
``WrongCroppingError`` that is *not* caught by the parent's retry loop, so
the failure propagates up and the inference loop can log + skip that index.
"""

from __future__ import annotations

import sys
from types import MethodType
from typing import TYPE_CHECKING, cast

import numpy as np

from miniworld.data.pipeline import get_chain_crop_indices
from miniworld.utils.crop import crop_spatial_segment_token

from ._modified_focus import (
    POLYATOMIC_IONS,
    SMALL_MOL_ENTITY_TYPES,
    SOLVENTS_AND_AIDS,
    WATER_CHEM_COMP,
    NoModifiedResidueError,
    select_modified_focus,
)


def _blocklist_residue_mask(cifmol: "CIFMolAttached") -> np.ndarray:
    """True at residue indices that are water / monatomic ions / polyatomic
    ions / common solvents and crystallization aids. Used to drop these
    residues from the inference crop so they are not part of the structure
    prediction or the reference written alongside it.
    """
    chain_entity_types = np.asarray(cifmol.chains.entity_type.value, dtype=object)
    chem_comp_ids = np.asarray(cifmol.residues.chem_comp_id.value, dtype=object)
    res_to_chain = np.asarray(cifmol.index_table.res_to_chain, dtype=np.int64)
    res_entity_types = chain_entity_types[res_to_chain]
    atom_to_res = np.asarray(cifmol.index_table.atom_to_res, dtype=np.int64)
    n_res = len(chem_comp_ids)
    n_atoms_per_res = np.bincount(atom_to_res, minlength=n_res)

    is_small = np.fromiter(
        (t in SMALL_MOL_ENTITY_TYPES for t in res_entity_types),
        dtype=bool, count=n_res,
    )
    is_water = np.fromiter(
        (str(c) in WATER_CHEM_COMP for c in chem_comp_ids),
        dtype=bool, count=n_res,
    )
    is_solvent_aid = np.fromiter(
        (str(c) in SOLVENTS_AND_AIDS for c in chem_comp_ids),
        dtype=bool, count=n_res,
    )
    is_polyion = np.fromiter(
        (str(c) in POLYATOMIC_IONS for c in chem_comp_ids),
        dtype=bool, count=n_res,
    )
    is_monoion = is_small & (n_atoms_per_res == 1)

    return is_water | is_solvent_aid | is_polyion | is_monoion

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


def _modified_focus_get_crop_indices(
    self,
    cifmol: CIFMolAttached,
    chain_ids: list[str],  # noqa: ARG001 — signature parity with the parent; ignored
    max_tokens: int,
    max_atoms: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Drop-in replacement for ``BioMolData.get_crop_indices`` (inference).

    The bias-chain pivot in the parent is replaced by a priority pick over
    the entire cifmol (mod protein → mod NA → small molecule, with water,
    monatomic ions, and AlphaFold3-listed crystallization aids excluded).

    Zero-length crops raise the *parent's* ``WrongCroppingError`` so the
    parent's retry-on-error loop still recognises it (each dataloader file
    defines its own ``WrongCroppingError`` class).
    """
    focus, _tag = select_modified_focus(cifmol, rng)

    segment_size = int(
        rng.integers(
            self.config.crop_config.min_segment_size,
            self.config.crop_config.max_segment_size + 1,
        ),
    )

    atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(cifmol)

    crop_indices = crop_spatial_segment_token(
        cifmol,
        focus,
        tokens_to_res=token_to_residue_idx_map,
        segment_size=segment_size,
        max_tokens=max_tokens,
        max_atoms=max_atoms,
    )

    blocklist = _blocklist_residue_mask(cifmol)
    if blocklist.any():
        crop_indices = crop_indices[~blocklist[crop_indices]]

    chain_id_to_crop_indices = get_chain_crop_indices(
        cifmol=cifmol,
        crop_indices=crop_indices,
    )

    crop_indices = cast("np.ndarray", crop_indices)

    if crop_indices.shape[0] == 0:
        parent_mod = sys.modules[type(self).__module__]
        wrong_cls = getattr(parent_mod, "WrongCroppingError", ValueError)
        msg = f"Failed to crop {cifmol.id} around modified-residue focus."
        raise wrong_cls(msg)

    token_mask = np.isin(token_to_residue_idx_map, crop_indices)
    cropped_token_indices = np.where(token_mask)[0]
    cropped_token_to_residue_idx_map = token_to_residue_idx_map[token_mask]

    max_res = token_to_residue_idx_map.max() + 1
    lookup = np.full(max_res, -1)
    lookup[crop_indices] = np.arange(len(crop_indices))
    cropped_token_to_residue_idx_map_reindexed = lookup[
        cropped_token_to_residue_idx_map
    ]

    atom_mask = np.isin(atom_to_token_idx_map, cropped_token_indices)
    cropped_atom_to_token_idx_map = atom_to_token_idx_map[atom_mask]

    max_token = atom_to_token_idx_map.max() + 1
    lookup_token = np.full(max_token, -1)
    lookup_token[cropped_token_indices] = np.arange(len(cropped_token_indices))
    cropped_atom_to_token_idx_map_reindexed = lookup_token[
        cropped_atom_to_token_idx_map
    ]

    return (
        crop_indices,
        chain_id_to_crop_indices,
        cropped_atom_to_token_idx_map_reindexed,
        cropped_token_to_residue_idx_map_reindexed,
        focus,
    )


def apply_modified_focus(dataset) -> None:
    """Swap the dataset's ``get_crop_indices`` for the modified-focus version.

    Works on either ``dataloader.BioMolData`` or ``dataloader_explicit.BioMolData``
    (or any subclass with the same method signature). Mutates the instance in
    place; the underlying class definition is untouched, so other instances
    (e.g. training datasets) are unaffected.
    """
    dataset.get_crop_indices = MethodType(_modified_focus_get_crop_indices, dataset)


__all__ = [
    "NoModifiedResidueError",
    "apply_modified_focus",
]
