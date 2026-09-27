"""Shared reference-coordinate repair for training and inference.

Complete model coordinates retain their existing meaning. Incomplete components
are regenerated as a whole with RDKit, never spliced between coordinate frames.
Unlike AF3's RDKit-first policy, this is a fallback for existing MiniWorld data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from miniworld.data.mols import CCDMol, FragmentedCCDMol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceCoordinates:
    """Finite coordinates, validity, and the selected coordinate source."""

    pos: np.ndarray
    mask: np.ndarray
    source: str


def parse_coordinates(raw: np.ndarray) -> ReferenceCoordinates:
    """Preserve validity before replacing missing values with finite padding."""
    values = np.array(raw, dtype=object, copy=True)
    values[(values == "?") | (values == ".") | (values == None)] = np.nan  # noqa: E711
    xyz = values.astype(np.float32)
    mask = np.isfinite(xyz).all(axis=-1)
    return ReferenceCoordinates(np.where(mask[:, None], xyz, 0.0), mask, "model")


def augment_reference(
    pos: np.ndarray,
    mask: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Center on valid atoms only; masked atoms remain finite zero padding."""
    if not np.any(mask):
        return np.zeros_like(pos, dtype=np.float32)
    center = pos[mask].mean(axis=0)
    return np.where(mask[:, None], (pos - center) @ rotation + translation, 0.0)


def _topology(mol: CCDMol | FragmentedCCDMol) -> tuple:
    atoms = mol.atoms
    names = tuple(str(x) for x in atoms.id.value)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate CCD atom names")
    elements = tuple(str(x) for x in atoms.element.value)
    charges = tuple(str(x) for x in atoms.charge.value)
    stereo = tuple(str(x) for x in atoms.stereo.value)
    edge = atoms.bond_type
    orders = np.asarray(edge.value).reshape(-1)
    bond_stereo = np.asarray(atoms.bond_stereo.value).reshape(-1)
    bonds = {}
    for i, j, order, st in zip(edge.src, edge.dst, orders, bond_stereo, strict=True):
        i, j = sorted((int(i), int(j)))
        value = (str(order), str(st))
        if (i, j) in bonds and bonds[i, j] != value:
            raise ValueError("Conflicting CCD bond definitions")
        bonds[i, j] = value
    return (
        names,
        elements,
        charges,
        stereo,
        tuple((i, j, *v) for (i, j), v in sorted(bonds.items())),
    )


@lru_cache(maxsize=1024)
def _rdkit_conformer(component: str, topology: tuple) -> np.ndarray | None:
    """Bounded, deterministic generation; successful and failed attempts cache."""
    # Missing RDKit is an installation error, not a chemical-generation failure.
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdCIPLabeler

    names, elements, charges, stereo, bonds = topology
    try:
        if len(names) > 512:
            raise ValueError("CCD exceeds the 512-atom fallback budget")
        rw = Chem.RWMol()
        for element, charge, st in zip(elements, charges, stereo, strict=True):
            atom = Chem.Atom(element)
            if charge in {"?", "."}:
                raise ValueError("Unknown formal charge")
            atom.SetFormalCharge(int(charge))
            if st in {"R", "S"}:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif st not in {"N", "?", "."}:
                raise ValueError(f"Unsupported atom stereo {st}")
            rw.AddAtom(atom)
        bond_types = {
            "SING": Chem.BondType.SINGLE,
            "DOUB": Chem.BondType.DOUBLE,
            "TRIP": Chem.BondType.TRIPLE,
            "AROM": Chem.BondType.AROMATIC,
        }
        for i, j, order, _ in bonds:
            rw.AddBond(i, j, bond_types[order])
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
        for atom, expected in zip(mol.GetAtoms(), stereo, strict=True):
            if expected in {"R", "S"}:
                if not atom.HasProp("_CIPCode"):
                    raise ValueError("Cannot resolve CCD tetrahedral stereochemistry")
                if atom.GetProp("_CIPCode") != expected:
                    atom.InvertChirality()
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
        for i, j, order, st in bonds:
            if st in {"E", "Z"}:
                if order != "DOUB":
                    raise ValueError("E/Z annotation on a non-double bond")
                left = [
                    a for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() != j
                ]
                right = [
                    a for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != i
                ]
                bond = mol.GetBondBetweenAtoms(i, j)
                # Use explicit neighbor-relative CIS/TRANS, then ask RDKit's
                # CIP implementation which absolute E/Z label it represents.
                # _CIPRank is private and absent in newer RDKit versions.
                bond.SetStereoAtoms(left[0].GetIdx(), right[0].GetIdx())
                bond.SetStereo(Chem.BondStereo.STEREOTRANS)
                rdCIPLabeler.AssignCIPLabels(mol)
                if not bond.HasProp("_CIPCode"):
                    raise ValueError("Cannot resolve CCD double-bond stereochemistry")
                if bond.GetProp("_CIPCode") != st:
                    bond.SetStereo(Chem.BondStereo.STEREOCIS)
                    bond.ClearProp("_CIPCode")
                    rdCIPLabeler.AssignCIPLabels(mol)
            elif st not in {"N", "?", "."}:
                raise ValueError(f"Unsupported bond stereo {st}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = (
            int.from_bytes(
                hashlib.sha256(repr(topology).encode()).digest()[:4], "little"
            )
            & 0x7FFFFFFF
        )
        params.numThreads = 1
        params.maxIterations = 200
        if hasattr(params, "timeout"):
            params.timeout = 5
        if AllChem.EmbedMolecule(mol, params) < 0:
            raise ValueError("RDKit embedding failed")
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
        for i, expected in enumerate(stereo):
            atom = mol.GetAtomWithIdx(i)
            if expected in {"R", "S"} and (
                not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != expected
            ):
                raise ValueError("Generated conformer violates CCD atom stereochemistry")
        for i, j, _, st in bonds:
            if st in {"E", "Z"}:
                bond = mol.GetBondBetweenAtoms(i, j)
                if not bond.HasProp("_CIPCode") or bond.GetProp("_CIPCode") != st:
                    raise ValueError(
                        "Generated conformer violates CCD bond stereochemistry"
                    )
        pos = np.asarray(
            mol.GetConformer().GetPositions()[: len(names)], dtype=np.float32
        )
        if not np.isfinite(pos).all():
            raise ValueError("Non-finite generated conformer")
        pos.setflags(write=False)
        return pos
    except (ValueError, RuntimeError, KeyError, IndexError, OverflowError) as exc:
        logger.warning(
            "Reference conformer fallback failed for %s: %s; missing atoms remain masked",
            component,
            exc,
        )
        return None


def resolve_reference(
    mol: CCDMol | FragmentedCCDMol, component: str
) -> ReferenceCoordinates:
    """Resolve a full CCD component before any cropping or atom removal."""
    original = parse_coordinates(mol.atoms.model_xyz.value)
    if original.mask.all():
        return original
    try:
        topology = _topology(mol)
    except (KeyError, ValueError) as exc:
        logger.warning(
            "Incomplete CCD topology for %s: %s; missing atoms remain masked",
            component,
            exc,
        )
        return ReferenceCoordinates(original.pos, original.mask, "model_masked")
    generated = _rdkit_conformer(component, topology)
    if generated is None:
        return ReferenceCoordinates(original.pos, original.mask, "model_masked")
    return ReferenceCoordinates(
        generated.copy(), np.ones(len(generated), dtype=bool), "rdkit"
    )


def map_reference(
    mol: CCDMol | FragmentedCCDMol,
    reference: ReferenceCoordinates,
    atom_names: np.ndarray,
) -> ReferenceCoordinates:
    """Map by atom name, preserving cropped/reordered caller atom order."""
    lookup = {str(name): i for i, name in enumerate(mol.atoms.id.value)}
    pos = np.zeros((len(atom_names), 3), dtype=np.float32)
    mask = np.zeros(len(atom_names), dtype=bool)
    for i, name in enumerate(atom_names):
        j = lookup.get(str(name))
        if j is not None:
            pos[i], mask[i] = reference.pos[j], reference.mask[j]
    return ReferenceCoordinates(pos, mask, reference.source)
