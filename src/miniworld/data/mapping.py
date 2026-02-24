from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class MoleculeType(Enum):
    """Enumeration of biological polymer types."""

    ANTIBODY = "antibody"  # including nanobody, scFv etc.
    PROTEIN = "protein"
    DPROTEIN = "dprotein"  # D-amino acid protein
    RNA = "rna"
    DNA = "dna"
    NA = "na"  # non-specific nucleic acid
    LIGAND = "ligand"
    BRANCHED = "branched"  # for branched polymers like glycans etc.


_atom_mapping = {
    "H": 0,
    "HE": 1,
    "LI": 2,
    "BE": 3,
    "B": 4,
    "C": 5,
    "N": 6,
    "O": 7,
    "F": 8,
    "NE": 9,
    "NA": 10,
    "MG": 11,
    "AL": 12,
    "SI": 13,
    "P": 14,
    "S": 15,
    "CL": 16,
    "AR": 17,
    "K": 18,
    "CA": 19,
    "SC": 20,
    "TI": 21,
    "V": 22,
    "CR": 23,
    "MN": 24,
    "FE": 25,
    "CO": 26,
    "NI": 27,
    "CU": 28,
    "ZN": 29,
    "GA": 30,
    "GE": 31,
    "AS": 32,
    "SE": 33,
    "BR": 34,
    "KR": 35,
    "RB": 36,
    "SR": 37,
    "Y": 38,
    "ZR": 39,
    "NB": 40,
    "MO": 41,
    "TC": 42,
    "RU": 43,
    "RH": 44,
    "PD": 45,
    "AG": 46,
    "CD": 47,
    "IN": 48,
    "SN": 49,
    "SB": 50,
    "TE": 51,
    "I": 52,
    "XE": 53,
    "CS": 54,
    "BA": 55,
    "LA": 56,
    "CE": 57,
    "PR": 58,
    "ND": 59,
    "PM": 60,
    "SM": 61,
    "EU": 62,
    "GD": 63,
    "TB": 64,
    "DY": 65,
    "HO": 66,
    "ER": 67,
    "TM": 68,
    "YB": 69,
    "LU": 70,
    "HF": 71,
    "TA": 72,
    "W": 73,
    "RE": 74,
    "OS": 75,
    "IR": 76,
    "PT": 77,
    "AU": 78,
    "HG": 79,
    "TL": 80,
    "PB": 81,
    "BI": 82,
    "PO": 83,
    "AT": 84,
    "RN": 85,
    "FR": 86,
    "RA": 87,
    "AC": 88,
    "TH": 89,
    "PA": 90,
    "U": 91,
    "NP": 92,
    "PU": 93,
    "AM": 94,
    "CM": 95,
    "BK": 96,
    "CF": 97,
    "ES": 98,
    "FM": 99,
    "MD": 100,
    "NO": 101,
    "LR": 102,
    "RF": 103,
    "DB": 104,
    "SG": 105,
    "BH": 106,
    "HS": 107,
    "MT": 108,
    "DS": 109,
    "RG": 110,
    "CN": 111,
    "NH": 112,
    "FL": 113,
    "MC": 114,
    "LV": 115,
    "TS": 116,
    "OG": 117,
    "X": 118,  # for unknown atoms
    "D": 119,  # for deuterium
}

_entity_tag_to_idx_mapping = {
    "A": 0,  # MoleculeType.ANTIBODY,
    "P": 1,  # MoleculeType.PROTEIN,
    "Q": 2,  # MoleculeType.DPROTEIN,
    "R": 3,  # MoleculeType.RNA,
    "D": 4,  # MoleculeType.DNA,
    "N": 5,  # MoleculeType.NA,
    "L": 6,  # MoleculeType.LIGAND,
    "B": 7,  # MoleculeType.BRANCHED,
    "X": 6,  # unknown molecule type treated as ligand
}

_entity_idx_to_type_mapping = {
    0: MoleculeType.ANTIBODY,
    1: MoleculeType.PROTEIN,
    2: MoleculeType.DPROTEIN,
    3: MoleculeType.RNA,
    4: MoleculeType.DNA,
    5: MoleculeType.NA,
    6: MoleculeType.LIGAND,
    7: MoleculeType.BRANCHED,
}

CANONICAL_CHEMCOMPS: set[str] = {
    # amino acids
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    # nucleotides
    "A",
    "U",
    "G",
    "C",
    "DA",
    "DT",
    "DG",
    "DC",
}


class AtomMapping:
    """Mapping of atom symbols to their corresponding indices."""

    def __init__(self) -> None:
        self._atom_to_index = _atom_mapping
        self._index_to_atom = {v: k for k, v in _atom_mapping.items()}

    def atom_to_index(self, atom: list[str] | np.ndarray) -> np.ndarray:
        """Convert an atom symbol to its corresponding index."""
        return np.array([self._atom_to_index.get(a, -1) for a in atom])

    def index_to_atom(self, index: list[int] | np.ndarray) -> np.ndarray:
        """Convert an index back to its corresponding atom symbol."""
        return np.array([self._index_to_atom.get(i, "X") for i in index])


class BaseResidueView:
    """Base view class for residue-to-index mappings."""

    GAP_INDEX: ClassVar[int] = 31

    def __init__(self, mapping_table: ResidueMapping) -> None:
        """Initialize view with a reference to the parent mapping table."""
        self._table = mapping_table

    def map(self, residues: Sequence[str] | np.ndarray) -> np.ndarray:
        """Map a sequence of residue symbols into integer indices."""
        if isinstance(residues, str):
            residues = [residues]
        # Use np.fromiter for efficient conversion
        return np.fromiter((self._map_single(r) for r in residues), dtype=np.int32)

    def _map_single(self, residue: str) -> int:
        """Map a single residue symbol to an integer index (to be implemented)."""
        raise NotImplementedError


class ProteinView(BaseResidueView):
    """Mapping view for protein amino acids."""

    def _map_single(self, aa: str) -> int:
        if aa == "-":
            return self.GAP_INDEX
        return self._table.AA2NUM.get(aa, self._table.AA_UNKNOWN)


class RNAView(BaseResidueView):
    """Mapping view for RNA nucleotides."""

    def _map_single(self, base: str) -> int:
        if base == "-":
            return self.GAP_INDEX
        return self._table.RNA2NUM.get(base, self._table.RNA_UNKNOWN)


class DNAView(BaseResidueView):
    """Mapping view for DNA nucleotides."""

    def _map_single(self, base: str) -> int:
        if base == "-":
            return self.GAP_INDEX
        return self._table.DNA2NUM.get(base, self._table.DNA_UNKNOWN)


class LigandView(BaseResidueView):
    """Mapping view for ligands or non-polymeric residues."""

    def _map_single(self, _: str) -> int:
        return self._table.LIGAND_INDEX


class ResidueMapping:
    """Unified residue-to-index mapping system providing specialized views."""

    # --- Protein constants ---
    NUM2AA: ClassVar[list[str]] = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
    ]
    AA2NUM: ClassVar[dict[str, int]] = {x: i for i, x in enumerate(NUM2AA)}
    AA2NUM.update(
        {
            "B": 3,  # Asx (D/N)
            "J": 20,  # X
            "O": 20,  # Pyrrolysine
            "U": 4,  # Selenocysteine
            "Z": 6,  # Glx (E/Q)
        },
    )
    AA_UNKNOWN: ClassVar[int] = 20

    # --- RNA constants ---
    RNA2NUM: ClassVar[dict[str, int]] = {"A": 21, "U": 22, "G": 23, "C": 24}
    RNA_UNKNOWN: ClassVar[int] = 25

    # --- DNA constants ---
    DNA2NUM: ClassVar[dict[str, int]] = {"A": 26, "T": 27, "G": 28, "C": 29}
    DNA_UNKNOWN: ClassVar[int] = 30

    # --- Special tokens ---
    LIGAND_INDEX: ClassVar[int] = 20

    MAX_INDEX: ClassVar[int] = 31  # Including gap

    def __init__(self) -> None:
        """Initialize all mapping views for different polymer types."""
        self.protein = ProteinView(self)
        self.rna = RNAView(self)
        self.dna = DNAView(self)
        self.ligand = LigandView(self)

    def get_view(self, polymer_type: MoleculeType) -> BaseResidueView:  # noqa: PLR0911
        """Return the corresponding mapping view for the given polymer type."""
        match polymer_type:
            case MoleculeType.ANTIBODY:
                return self.protein
            case MoleculeType.PROTEIN:
                return self.protein
            case MoleculeType.DPROTEIN:
                return self.protein
            case MoleculeType.RNA:
                return self.rna
            case MoleculeType.DNA:
                return self.dna
            case MoleculeType.NA:
                return self.rna
            case _:
                return self.ligand


class EntityMapping:
    """Mapping of entity types to MoleculeType enums."""

    def __init__(self) -> None:
        self._entity_tag_to_idx = _entity_tag_to_idx_mapping
        self._entity_idx_to_type = _entity_idx_to_type_mapping

    def tag_to_idx(self, entities: Sequence[str] | np.ndarray) -> np.ndarray:
        """Map a sequence of entity type symbols into MoleculeType enums."""
        if isinstance(entities, str):
            entities = [entities]
        return np.array(
            [self._entity_tag_to_idx.get(e, MoleculeType.LIGAND) for e in entities]
        )

    def idx_to_type(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Convert a sequence of indices back to MoleculeType enums."""
        if isinstance(indices, int):
            indices = [indices]
        return np.array(
            [self._entity_idx_to_type.get(i, MoleculeType.LIGAND) for i in indices]
        )
