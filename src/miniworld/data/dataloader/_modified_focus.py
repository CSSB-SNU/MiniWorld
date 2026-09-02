"""Modified-residue focus selection for inference-time cropping.

The training dataloaders pick a random pivot atom inside the bias chain(s)
and let the spatial cropper expand from there. For inference on a dataset
trimmed to structures with modified residues / glycans, that random pivot
often misses the modification, leaving the crop centered elsewhere.

This module provides a priority-based picker that scans the *entire* cifmol
and returns a focus xyz lying on a modified residue, with the priority order:

    1. modified protein residue (entity_type in protein, chem_comp not standard AA)
    2. modified nucleic acid    (entity_type in NA,      chem_comp not standard NA)
    3. small molecule           (entity_type LIGAND/BRANCHED, excluding water,
                                 monatomic ions, and common crystallization aids)

Glycans are stored as separate BRANCHED entities in mmCIF and therefore land
in bucket (3), not bucket (1).

The crystallization-aid blocklist follows AlphaFold3 SI Chapter 2.5.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


# cifmol.chains.entity_type.value yields CIF _entity_poly.type strings for
# polymers, and CIF _entity.type strings ("non-polymer", "branched") otherwise.
PROTEIN_ENTITY_TYPES: frozenset[str] = frozenset({
    "polypeptide(L)",
    "polypeptide(D)",
})
NA_ENTITY_TYPES: frozenset[str] = frozenset({
    "polyribonucleotide",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
})
SMALL_MOL_ENTITY_TYPES: frozenset[str] = frozenset({
    "non-polymer",
    "branched",
    "other",
})

STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})
STANDARD_NA: frozenset[str] = frozenset({
    "DA", "DC", "DG", "DT", "A", "C", "G", "U",
})
WATER_CHEM_COMP: frozenset[str] = frozenset({
    "HOH", "DOD", "H2O", "WAT", "TIP", "TIP3", "TIP3P", "TP3", "SOL",
})
# Kept in sync with scripts/make_monomer_mod_glycan_ligand_set.py:
# AF3 SI Chapter 2.5.4 base list + extended buffers / solvents / detergents.
SOLVENTS_AND_AIDS: frozenset[str] = frozenset({
    # AF3 SI base list
    "SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS",
    "PGE", "PG4", "FMT", "EPE", "MPD", "MES", "CD",  "IOD",
    # Organic buffers / pH aids / chelators
    "CIT", "FLC", "TLA", "TAR", "MLI", "MLA", "MAE", "BCT",
    "BCN", "BTB", "CHT", "IMD", "TRT", "ACY",
    # PEG / polyol variants
    "1PE", "PG5", "PG6", "PG0", "P33", "P4G", "PE3", "PE4",
    # Alcohols / diols / common solvents
    "IPA", "EOH", "MOH", "MRD", "BU3", "DIO", "ETX",
    # Reducing agents
    "BME", "DTT", "DTU",
    # Common detergents
    "LDA", "BNG", "C8E", "DMU", "B7G",
})
# Inorganic polyatomic ions and phosphate-mimic ions. Single-atom ions are
# already handled by the n_atoms_per_res == 1 check.
#
# Every code below was checked against the chemical component dictionary
# (preprocessed_CCD_20260826.lmdb) on 2026-08-31. An earlier revision of this
# list was written from chemical formulas rather than CCD codes, which put 27
# wrong entries in it:
#
#   19 organic molecules whose code merely looks like an ion formula --
#     PPI is PROPANOIC ACID (pyrophosphate is PPV/DPO), CL1/CL2 are chlorophyll
#     a, IO3 is iophenoxic acid, IO4 a pyrazolo compound, BF3 an isoindole, BO2
#     a boronic-acid inhibitor, SEO is 2-mercaptoethanol, NCS a spiro-naphthalene
#     (thiocyanate is SCN), CNO/CRO/CR4/MNO/RE4/TC4/HPO/RUS organics, AO3 is
#     ALLOSAMIDIN and PPS is PAPS -- both genuine ligands that were being
#     excluded from the small_molecule pool.
#
#    8 codes that are not in the CCD at all: BRO, BRO3, CLO, CLO3, CLO4, N3,
#     S2O3, S2O8. The real codes for two of them are AZI (azide) and THJ
#     (thiosulfate), which are listed below.
#
# Note MO3/MO4 are hydrated magnesium ions, not molybdate; molybdate is MOO.
POLYATOMIC_IONS: frozenset[str] = frozenset({
    # Nitrogen-based anions
    "NO3", "NO2", "AZI",
    # Cyanide / thiocyanate / cyano and ammine complexes
    "CN", "SCN", "TCN", "NCO",
    # Sulfur oxyanions (SO4 already in aids)
    "SO3", "SUL", "THJ",
    # Phosphorus oxyanions (PO4 already in aids)
    "PO3", "PPV", "DPO", "POP", "2HP", "PI",
    # Boron-fluorine / boron oxyanions
    "BF4", "BO3", "BO4",
    # Phosphate-mimic transition-state analogs
    "BEF", "ALF", "MGF", "AF3",
    # Transition-metal oxoanions
    "WO4", "WO3", "VO4", "VO3", "REO", "RUO", "MOO",
    # Hydrated / multinuclear metal ions
    "MO3", "MO4", "NI2", "CUA",
    # Misc inorganic
    "OH", "NH4", "PER", "PEO", "ARS",
})

# Monatomic ions whose CCD code is not a bare element symbol -- charge-state and
# variant codes. ``is_ion`` catches these structurally via n_atoms_per_res == 1,
# so this set exists for code-level classification elsewhere (see
# scripts/dataset/ligand_classes.py), not for the focus pools below.
# Every code verified as exactly one heavy atom against the CCD, 2026-08-31.
# Deliberately NOT here: CO2 is carbon dioxide, RU7 is para-cymene ruthenium
# chloride, ZN2 is not a CCD code -- the same look-alike trap as above.
MONATOMIC_ION_CODES: frozenset[str] = frozenset({
    "FE2", "CU1", "3CO", "MN3", "AU3", "IR3", "YT3", "Y1", "U1",
    "F", "W", "O", "OS4", "PT4", "RB",
})


class NoModifiedResidueError(Exception):
    """Raised when no residue in any priority bucket has finite coordinates."""


def _cifmol_id(cifmol: CIFMolAttached) -> str:
    """Best-effort extraction of a printable cifmol id for log lines."""
    raw = getattr(cifmol, "id", None)
    if raw is None:
        return "?"
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
        return str(raw[0])
    return str(raw)


def select_modified_focus(
    cifmol: CIFMolAttached,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Pick a focus xyz centered on a modified residue.

    Returns
    -------
    focus_xyz : ndarray, shape (3,)
        Coordinates of a random finite-position atom on the chosen modified residue.
    priority_tag : str
        One of ``"mod_protein"``, ``"mod_nucleic_acid"``, ``"small_molecule"``,
        identifying which bucket the focus was drawn from.

    Raises
    ------
    NoModifiedResidueError
        If every priority bucket is empty (no candidate residue with at least
        one finite-coordinate atom).
    """
    chain_entity_types = np.asarray(cifmol.chains.entity_type.value, dtype=object)
    chem_comp_ids = np.asarray(cifmol.residues.chem_comp_id.value, dtype=object)
    res_to_chain = np.asarray(cifmol.index_table.res_to_chain, dtype=np.int64)
    res_entity_types = chain_entity_types[res_to_chain]  # (N_res,)

    atom_to_res = np.asarray(cifmol.index_table.atom_to_res, dtype=np.int64)
    xyz = np.asarray(cifmol.atoms.xyz.value)
    finite = np.isfinite(xyz).all(axis=-1)

    n_res = len(chem_comp_ids)
    n_atoms_per_res = np.bincount(atom_to_res, minlength=n_res)
    n_finite_per_res = np.bincount(atom_to_res[finite], minlength=n_res)

    is_protein = np.fromiter(
        (t in PROTEIN_ENTITY_TYPES for t in res_entity_types), dtype=bool, count=n_res,
    )
    is_na = np.fromiter(
        (t in NA_ENTITY_TYPES for t in res_entity_types), dtype=bool, count=n_res,
    )
    is_small = np.fromiter(
        (t in SMALL_MOL_ENTITY_TYPES for t in res_entity_types), dtype=bool, count=n_res,
    )

    is_std_aa = np.fromiter(
        (str(c) in STANDARD_AA for c in chem_comp_ids), dtype=bool, count=n_res,
    )
    is_std_na = np.fromiter(
        (str(c) in STANDARD_NA for c in chem_comp_ids), dtype=bool, count=n_res,
    )
    is_water = np.fromiter(
        (str(c) in WATER_CHEM_COMP for c in chem_comp_ids), dtype=bool, count=n_res,
    )
    is_solvent_aid = np.fromiter(
        (str(c) in SOLVENTS_AND_AIDS for c in chem_comp_ids), dtype=bool, count=n_res,
    )
    is_polyion = np.fromiter(
        (str(c) in POLYATOMIC_IONS for c in chem_comp_ids), dtype=bool, count=n_res,
    )
    # single-atom non-polymer residue = monatomic ion (NA, MG, ZN, CL, ...)
    is_ion = is_small & (n_atoms_per_res == 1)

    has_finite_atom = n_finite_per_res > 0

    pools: list[tuple[str, np.ndarray]] = [
        ("mod_protein", is_protein & ~is_std_aa & has_finite_atom),
        ("mod_nucleic_acid", is_na & ~is_std_na & has_finite_atom),
        ("small_molecule",
            is_small & ~is_water & ~is_ion & ~is_solvent_aid & ~is_polyion
            & has_finite_atom),
    ]

    for tag, mask in pools:
        candidate_res = np.flatnonzero(mask)
        if candidate_res.size == 0:
            continue
        res_idx = int(rng.choice(candidate_res))
        atom_pool = np.flatnonzero((atom_to_res == res_idx) & finite)
        atom_idx = int(rng.choice(atom_pool))
        focus = np.asarray(xyz[atom_idx], dtype=np.float64)
        return focus, tag

    msg = f"No modified residue found in cifmol {_cifmol_id(cifmol)}."
    raise NoModifiedResidueError(msg)
