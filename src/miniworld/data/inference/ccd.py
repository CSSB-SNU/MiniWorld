"""CCD lookup helper for inference Batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from miniworld.data.io import load_raw_data
from miniworld.data.mols import CCDMol, FragmentedCCDMol
from miniworld.data.pipeline import fragment_ccdmol_all_merges


@dataclass(frozen=True)
class CCDResidue:
    """Per-residue topology pulled from a single CCD entry.

    Atom positions are taken from the canonical CCD model; they are used as
    the model's reference (``ReferenceFeatures.pos``) regardless of whether a
    real GT structure is available.
    """

    chemcomp_id: str
    atom_ids: np.ndarray         # (n_atoms,) object/str
    atom_elements: np.ndarray    # (n_atoms,) object/str
    atom_charges: np.ndarray     # (n_atoms,) float
    atom_xyz: np.ndarray         # (n_atoms, 3) float

    @property
    def n_atoms(self) -> int:
        return int(self.atom_xyz.shape[0])


class CCDLookup:
    """Cache of ``CCDResidue`` entries (and optional fragment dicts) keyed by chemcomp id."""

    def __init__(self, ccd_db_path: Path) -> None:
        self.ccd_db_path = Path(ccd_db_path)
        self._residue_cache: dict[str, CCDResidue] = {}
        self._fragments_cache: dict[str, dict[int, FragmentedCCDMol]] = {}
        self._ccdmol_cache: dict[str, CCDMol] = {}

    def __getitem__(self, chemcomp_id: str) -> CCDResidue:
        if chemcomp_id in self._residue_cache:
            return self._residue_cache[chemcomp_id]
        ccdmol = self._load_ccdmol(chemcomp_id)
        residue = _ccdmol_to_residue(chemcomp_id, ccdmol)
        self._residue_cache[chemcomp_id] = residue
        return residue

    def fragments(self, chemcomp_id: str) -> dict[int, FragmentedCCDMol]:
        """Return ``{merge_level: FragmentedCCDMol}`` for the CCD.

        Mirrors the dataloader's ``FragmentedCCDMolCache``: keys are
        ``0 .. max_effective_merge + 1`` where 0 is atomize and the largest key
        is a single fragment covering the whole CCD.
        """
        if chemcomp_id in self._fragments_cache:
            return self._fragments_cache[chemcomp_id]
        ccdmol = self._load_ccdmol(chemcomp_id)
        fragments = fragment_ccdmol_all_merges(ccdmol)
        self._fragments_cache[chemcomp_id] = fragments
        return fragments

    def _load_ccdmol(self, chemcomp_id: str) -> CCDMol:
        if chemcomp_id in self._ccdmol_cache:
            return self._ccdmol_cache[chemcomp_id]
        raw = load_raw_data(chemcomp_id, self.ccd_db_path)
        if raw is None:
            msg = f"CCD entry {chemcomp_id!r} not found in {self.ccd_db_path}."
            raise KeyError(msg)
        ccdmol = CCDMol.from_bytes(raw)
        self._ccdmol_cache[chemcomp_id] = ccdmol
        return ccdmol


def _ccdmol_to_residue(chemcomp_id: str, ccdmol: CCDMol) -> CCDResidue:
    atom_ids = np.asarray(ccdmol.atoms.id.value)
    atom_elements = np.asarray(ccdmol.atoms.element.value)

    raw_xyz = np.asarray(ccdmol.atoms.model_xyz.value, dtype=object)
    missing = (raw_xyz == "?") | (raw_xyz == ".")
    raw_xyz[missing] = 0.0
    atom_xyz = raw_xyz.astype(np.float32, copy=False)
    if np.isnan(atom_xyz).any():
        atom_xyz = np.nan_to_num(atom_xyz, nan=0.0)

    raw_charge = np.asarray(ccdmol.atoms.charge.value)
    atom_charges = np.array(
        [float(c) if c not in {"?", "."} else 0.0 for c in raw_charge],
        dtype=np.float32,
    )

    return CCDResidue(
        chemcomp_id=chemcomp_id,
        atom_ids=atom_ids,
        atom_elements=atom_elements,
        atom_charges=atom_charges,
        atom_xyz=atom_xyz,
    )
