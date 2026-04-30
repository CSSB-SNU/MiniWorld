"""Fasta parsing for inference (header-aware, ligand/branched aware)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EntityType(str, Enum):
    """Entity types declared in the fasta header.

    Mirrors the values produced by the merged.fasta convention.
    """

    PROTEIN = "polypeptide(L)"
    RNA = "polyribonucleotide"
    DNA = "polydeoxyribonucleotide"
    NON_POLYMER = "non-polymer"
    BRANCHED = "branched"


# Standard 20 amino acids: 1-letter -> 3-letter CCD code.
_AA_1TO3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# RNA 1-letter == CCD code (canonical).
_RNA_1TO_CCD: dict[str, str] = {"A": "A", "U": "U", "G": "G", "C": "C"}

# DNA 1-letter -> 2-letter CCD code.
_DNA_1TO_CCD: dict[str, str] = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}


_HEADER_RE = re.compile(r"^>\s*(?P<id>\S+)\s*\|\s*(?P<type>[^|]+?)\s*\|\s*Chain:(?P<chain>\S+)\s*$")
_CCD_PAREN_RE = re.compile(r"\(([^()]+)\)")
_BRANCHED_BOND_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


def index_to_lowercase_letter(i: int) -> str:
    """0->'a', 25->'z', 26->'aa', 27->'ab', ... — used for free-form fasta."""
    if i < 0:
        msg = f"chain index must be non-negative, got {i}."
        raise ValueError(msg)
    out: list[str] = []
    n = i
    while True:
        out.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(out))


# Sequence-letter sets for entity-type auto-detection (free-form fasta).
_AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYBZUOJX-")
_RNA_LETTERS = set("ACGUNX-")
_DNA_LETTERS = set("ACGTNX-")


def _auto_detect_entity_type(body: str) -> EntityType:
    """Pick an entity type from the residue letters in ``body``.

    The heuristic mirrors the merged.fasta convention (parens = ligand /
    branched, RNA if it has a ``U``, DNA if it has a ``T``, else protein) and
    is only used when the fasta header doesn't carry an explicit
    ``| <type> |`` field.
    """
    has_paren = "(" in body or ")" in body
    if has_paren:
        if "|" in body or len(_CCD_PAREN_RE.findall(body)) > 1:
            return EntityType.BRANCHED
        return EntityType.NON_POLYMER

    chars = {c for c in body.upper() if not c.isspace()}
    if not chars:
        return EntityType.PROTEIN
    has_t = "T" in chars
    has_u = "U" in chars
    only_dna = chars.issubset(_DNA_LETTERS)
    only_rna = chars.issubset(_RNA_LETTERS)
    only_protein = chars.issubset(_AA_LETTERS)
    if only_rna and has_u and not has_t:
        return EntityType.RNA
    if only_dna and has_t and not has_u:
        return EntityType.DNA
    if only_protein:
        return EntityType.PROTEIN
    # Mixed / unknown: best effort — protein covers most non-CCD chains.
    return EntityType.PROTEIN


@dataclass
class ChainSpec:
    """One chain parsed from a single fasta file."""

    chain_index: int                  # 0-based chain index from the JSON spec
    chain_letter: str                 # chain letter declared via ``Chain:<X>`` in the header
    fasta_id: str                     # raw header id (e.g., "5CCX_A_A")
    entity_type: EntityType
    chemcomp_ids: list[str]           # per-residue CCD codes
    one_letter_seq: list[str]         # per-residue 1-letter (or 'X' for ligand) — used for MSA encoding
    branched_bonds: list[tuple[int, int]]  # 0-based residue index pairs, polymer/branched only


def parse_fasta_file(path: Path, chain_index: int) -> ChainSpec:
    """Parse a single-record fasta file into a ``ChainSpec``.

    Two header conventions are accepted:

    1. **Strict** ``"> name | <entity_type> | Chain:<X>"`` — used by the
       roundtrip dump and the merged.fasta layout. The chain letter and
       entity_type come straight from the header.
    2. **Free-form** ``">anything you want"`` — chain letter is derived
       from ``chain_index`` (``0->'a'``, ``25->'z'``, then ``aa, ab, ...``)
       and entity_type is auto-detected from the residue letters in the body
       via :func:`_auto_detect_entity_type`.
    """
    text = Path(path).read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        msg = f"Empty fasta file: {path}."
        raise ValueError(msg)
    if not lines[0].startswith(">"):
        msg = f"Fasta {path} must start with a '>' header line."
        raise ValueError(msg)

    headers_after = [ln for ln in lines[1:] if ln.startswith(">")]
    if headers_after:
        msg = f"Fasta {path} contains more than one record; expected exactly one."
        raise ValueError(msg)
    body = "".join(lines[1:])

    header_match = _HEADER_RE.match(lines[0])
    if header_match is not None:
        fasta_id = header_match.group("id")
        type_str = header_match.group("type").strip()
        chain_letter = header_match.group("chain").strip()
        try:
            entity_type = EntityType(type_str)
        except ValueError as e:
            valid = [t.value for t in EntityType]
            msg = (
                f"Unknown entity_type {type_str!r} in {path}; "
                f"expected one of {valid}."
            )
            raise ValueError(msg) from e
    else:
        # Free-form header: derive everything from order + body content.
        first_token = lines[0][1:].split()[0] if len(lines[0]) > 1 else ""
        fasta_id = first_token or f"chain_{chain_index}"
        chain_letter = index_to_lowercase_letter(chain_index)
        entity_type = _auto_detect_entity_type(body)

    # token_bond mirrors the dataloader's `atom_bonds_to_token_bonds`, which
    # extracts only `covale` struct_conn entries (non-canonical covalent links).
    # Standard polymer backbone bonds are *implicit* and not stored, so we
    # only emit explicit edges for branched chains here.
    if entity_type == EntityType.PROTEIN:
        ccds, one_letter = _parse_protein_body(body, path)
        bonds = []
    elif entity_type == EntityType.RNA:
        ccds, one_letter = _parse_rna_body(body, path)
        bonds = []
    elif entity_type == EntityType.DNA:
        ccds, one_letter = _parse_dna_body(body, path)
        bonds = []
    elif entity_type == EntityType.NON_POLYMER:
        ccds, one_letter = _parse_non_polymer_body(body, path)
        bonds = []
    elif entity_type == EntityType.BRANCHED:
        ccds, one_letter, bonds = _parse_branched_body(body, path)
    else:  # pragma: no cover
        msg = f"Unhandled entity_type {entity_type}."
        raise ValueError(msg)

    return ChainSpec(
        chain_index=chain_index,
        chain_letter=chain_letter,
        fasta_id=fasta_id,
        entity_type=entity_type,
        chemcomp_ids=ccds,
        one_letter_seq=one_letter,
        branched_bonds=bonds,
    )


def _parse_protein_body(body: str, path: Path) -> tuple[list[str], list[str]]:
    ccds = []
    one_letter = []
    for ch in body:
        ccd = _AA_1TO3.get(ch.upper())
        if ccd is None:
            msg = f"Unknown amino acid letter {ch!r} in {path}."
            raise ValueError(msg)
        ccds.append(ccd)
        one_letter.append(ch.upper())
    return ccds, one_letter


def _parse_rna_body(body: str, path: Path) -> tuple[list[str], list[str]]:
    ccds = []
    one_letter = []
    for ch in body:
        ccd = _RNA_1TO_CCD.get(ch.upper())
        if ccd is None:
            msg = f"Unknown RNA letter {ch!r} in {path}."
            raise ValueError(msg)
        ccds.append(ccd)
        one_letter.append(ch.upper())
    return ccds, one_letter


def _parse_dna_body(body: str, path: Path) -> tuple[list[str], list[str]]:
    ccds = []
    one_letter = []
    for ch in body:
        ccd = _DNA_1TO_CCD.get(ch.upper())
        if ccd is None:
            msg = f"Unknown DNA letter {ch!r} in {path}."
            raise ValueError(msg)
        ccds.append(ccd)
        one_letter.append(ch.upper())
    return ccds, one_letter


def _parse_non_polymer_body(body: str, path: Path) -> tuple[list[str], list[str]]:
    matches = _CCD_PAREN_RE.findall(body)
    if not matches:
        msg = f"non-polymer body in {path} must be of the form '(CCD)'; got {body!r}."
        raise ValueError(msg)
    if len(matches) != 1:
        msg = (
            f"non-polymer body in {path} must contain exactly one (CCD); "
            f"got {len(matches)}: {matches}."
        )
        raise ValueError(msg)
    ccd = matches[0].strip()
    return [ccd], ["X"]


def _parse_branched_body(body: str, path: Path) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    parts = body.split("|", 1)
    ccd_section = parts[0]
    bond_section = parts[1] if len(parts) > 1 else ""

    ccds = [m.strip() for m in _CCD_PAREN_RE.findall(ccd_section)]
    if not ccds:
        msg = f"branched body in {path} must contain at least one (CCD); got {body!r}."
        raise ValueError(msg)

    bonds: list[tuple[int, int]] = []
    for m in _BRANCHED_BOND_RE.finditer(bond_section):
        i, j = int(m.group(1)), int(m.group(2))
        if not (1 <= i <= len(ccds) and 1 <= j <= len(ccds)):
            msg = (
                f"branched bond ({i},{j}) in {path} out of range; "
                f"chain has {len(ccds)} residues."
            )
            raise ValueError(msg)
        a, b = (i - 1, j - 1)
        bonds.append((min(a, b), max(a, b)))

    one_letter = ["X"] * len(ccds)
    return ccds, one_letter, bonds
