"""Multi-chain (complex) template loading.

Supports two sources, mirroring the user's spec:

* **Option A** — a CIF file path (``ComplexTemplateSpec.cif``). Parsed via
  ``Bio.PDB.MMCIFParser`` (same library StructCooker uses in
  ``pipelines/transforms/cif_transforms.py``).
* **Option B** — an LMDB cif id (``ComplexTemplateSpec.cif_id``) looked up in
  ``InferenceSpec.cif_db`` via the dataloader's ``load_cifmol``.

Per-chain N/CA/C/CB extraction follows StructCooker's
``extract_backbone_indices_from_cifmol`` (in
``pipelines/instructions/template_instructions.py``): for each residue, find
N/CA/C/CB atoms; if CB is missing, fall back to CA.

Each ``ComplexTemplateSpec`` becomes a single ``ProteinTemplate`` slot whose
coordinates are in one rigid frame across the participating chains. Chains
not in ``chain_map`` contribute an empty (mask=False) slot so the slot count
stays consistent across the batch.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from miniworld.data.constants import ResidueMapping
from miniworld.data.io import load_cifmol
from miniworld.data.pipeline import ProteinTemplate

if TYPE_CHECKING:
    from .build import _ChainExpansion
    from .spec import ComplexTemplateSpec, InferenceSpec


_BACKBONE_ATOMS = ("N", "CA", "C", "CB")


# Reuse the same 3-letter <-> 1-letter table as the fasta parser.
_AA_3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _three_to_one(resname: str) -> str:
    """Tiny 3-letter -> 1-letter helper (replaces removed Bio.PDB.three_to_one)."""
    return _AA_3TO1.get(resname.upper(), "X")


# ---------------------------------------------------------------------------
# Backbone extraction (Option A: CIF file)
# ---------------------------------------------------------------------------


def _open_cif(cif_path: Path) -> tuple[str, Path | None]:
    """Return ``(path_to_use, tmp_to_delete)`` for ``Bio.PDB.MMCIFParser``.

    biopython's parser doesn't transparently handle ``.gz``; we decompress to
    a temp file first (StructCooker takes the same approach in
    ``cif_transforms.get_cif_data``).
    """
    suffix = cif_path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(cif_path, "rb") as src:
            data = src.read()
        tmp = Path(tempfile.mkstemp(suffix=".cif")[1])
        tmp.write_bytes(data)
        return str(tmp), tmp
    return str(cif_path), None


def _load_chain_backbone_from_cif(
    cif_path: Path,
    template_chain_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Parse a CIF file and return ``(n_res, 4, 3)`` N/CA/C/CB coords + 1-letter codes.

    ``template_chain_id`` follows the BioMolDB convention
    ``"<label_asym_id>_<operator_id>"`` (e.g. ``"A_1"``). Raw CIF files
    don't carry operator instances, so we strip the ``_<digits>`` suffix
    before matching biopython's ``chain.id`` (which we configure to expose
    ``label_asym_id`` via ``auth_chains=False``).

    Heteroatoms (residues whose hetflag != ' ') are skipped. CB falls back
    to CA when missing (matches StructCooker's pattern). ``.cif.gz`` is
    transparently decompressed to a temp file before parsing.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    # ``auth_chains=False`` makes ``chain.id`` use ``label_asym_id`` (the
    # mmCIF authoritative cif id), matching the convention we use in
    # ``ComplexTemplateSpec.chain_map`` values.
    parser = MMCIFParser(QUIET=True, auth_chains=False)
    parse_path, tmp_path = _open_cif(cif_path)
    try:
        structure = parser.get_structure("complex", parse_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    model = next(iter(structure))
    # Strip the operator suffix (StructCooker pattern: chain_id.split("_")[0])
    label_id = template_chain_id.split("_")[0]
    target_chain = None
    for chain in model:
        if chain.id == label_id:
            target_chain = chain
            break
    if target_chain is None:
        msg = (
            f"Chain {template_chain_id!r} (label_asym_id {label_id!r}) "
            f"not found in {cif_path}. Available chains: {[c.id for c in model]}"
        )
        raise KeyError(msg)

    residues = [r for r in target_chain if r.id[0] == " "]  # standard residues only
    n_res = len(residues)
    if n_res == 0:
        msg = f"Chain {template_chain_id!r} in {cif_path} has no standard residues."
        raise ValueError(msg)

    bb = np.full((n_res, 4, 3), np.nan, dtype=np.float32)
    one_letter: list[str] = []
    for ri, res in enumerate(residues):
        atoms = {atom.get_name(): atom.get_coord() for atom in res}
        for ai, name in enumerate(_BACKBONE_ATOMS):
            if name in atoms:
                bb[ri, ai] = atoms[name]
        # CB fallback to CA (matches StructCooker)
        if not np.isfinite(bb[ri, 3]).all() and np.isfinite(bb[ri, 1]).all():
            bb[ri, 3] = bb[ri, 1]
        one_letter.append(_three_to_one(res.resname))
    return bb, one_letter


# ---------------------------------------------------------------------------
# Backbone extraction (Option B: cif LMDB)
# ---------------------------------------------------------------------------


def _load_chain_backbone_from_lmdb(
    cif_db_path: Path,
    cif_id: str,
    template_chain_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Load a chain's CIFMolAttached and extract the same N/CA/C/CB array.

    ``cif_id`` is a BioMolDB key in ``"<pdb>_<assembly>_<model>_<alt>"`` form.
    ``template_chain_id`` is the **BioMolDB chain id** — the same string
    stored in ``cifmol.chains.chain_id``, which has the form
    ``"<label_asym_id>_<operator_id>"`` (e.g. ``"D_1"``, ``"A_2"``).
    """
    parts = cif_id.split("_")
    if len(parts) != 4:
        msg = (
            f"cif_id {cif_id!r} must be of the form "
            f"'<pdb_id>_<assembly_id>_<model_id>_<alt_id>'."
        )
        raise ValueError(msg)
    pdb_id, assembly_id, model_id, alt_id = parts
    cifmol = load_cifmol(cif_db_path, pdb_id.lower(), assembly_id, model_id, alt_id)
    chain_view = cifmol.chains[cifmol.chains.chain_id == template_chain_id]
    if len(chain_view) == 0:
        available = [str(c) for c in np.asarray(cifmol.chains.chain_id.value)]
        msg = (
            f"Chain {template_chain_id!r} not found in cif {cif_id!r}. "
            f"Available chain ids: {available}"
        )
        raise KeyError(msg)
    chain_cifmol = chain_view.extract()

    n_res = len(chain_cifmol.residues)
    bb = np.full((n_res, 4, 3), np.nan, dtype=np.float32)
    for ri in range(n_res):
        residue = chain_cifmol.residues[ri]
        atom_ids = np.asarray(residue.atoms.id.value)
        xyz = np.asarray(residue.atoms.xyz.value, dtype=np.float32)
        for ai, name in enumerate(_BACKBONE_ATOMS):
            mask = atom_ids == name
            if mask.any():
                bb[ri, ai] = xyz[mask][0]
        if not np.isfinite(bb[ri, 3]).all() and np.isfinite(bb[ri, 1]).all():
            bb[ri, 3] = bb[ri, 1]

    one_letter_arr = np.asarray(chain_cifmol.residues.one_letter_code_can.value)
    one_letter = [str(c) for c in one_letter_arr]
    return bb, one_letter


# ---------------------------------------------------------------------------
# ProteinTemplate construction
# ---------------------------------------------------------------------------


def _align_template_to_query(
    template_bb: np.ndarray,           # (T, 4, 3)
    template_one_letter: list[str],
    query_one_letter: list[str],
    *,
    where: str = "",
) -> tuple[np.ndarray, list[str]]:
    """Trim a template chain so its residues match the query chain length.

    Strict path: if the lengths already match, accept without checking
    sequence identity (handles modified residues etc.). Otherwise, fall back
    to a contiguous-substring search on the 1-letter sequences — covers the
    common case where the template chain is the full PDB chain and the query
    is a (prefix / suffix / interior) subsequence.
    """
    n_q = len(query_one_letter)
    n_t = len(template_one_letter)
    if n_t == n_q:
        return template_bb, list(template_one_letter)
    template_seq = "".join(template_one_letter)
    query_seq = "".join(query_one_letter)
    idx = template_seq.find(query_seq)
    if idx >= 0:
        return (
            template_bb[idx: idx + n_q],
            list(template_one_letter[idx: idx + n_q]),
        )
    msg = (
        f"{where}: cannot align template chain of length {n_t} to query "
        f"chain of length {n_q}. Sequence identity / substring match failed.\n"
        f"  query head:    {query_seq[:60]}...\n"
        f"  template head: {template_seq[:60]}..."
    )
    raise ValueError(msg)


def _make_complex_slot_for_chain(
    bb: np.ndarray,
    one_letter: list[str],
    template_id: int,
) -> ProteinTemplate:
    """Build a 1-slot ProteinTemplate from N/CA/C/CB coords + 1-letter sequence."""
    n_res = bb.shape[0]
    rm = ResidueMapping()
    res_type = rm.protein.map(np.array(one_letter, dtype=object)).astype(np.int32)
    cb_xyz = bb[:, 3, :]                  # (n_res, 3)
    bb_xyz_arr = bb[:, :3, :]             # (n_res, 3, 3) for N, CA, C
    cb_mask = np.isfinite(cb_xyz).all(axis=-1)
    bb_mask = np.isfinite(bb_xyz_arr).all(axis=(-1, -2))
    return ProteinTemplate._from_arrays(  # noqa: SLF001
        mask=np.array([True], dtype=bool),
        ids=np.full((1, n_res), fill_value=template_id, dtype=object),
        res_type=res_type[None, :],
        cb_xyz=cb_xyz[None, :, :],
        cb_mask=cb_mask[None, :],
        bb_xyz=bb_xyz_arr[None, :, :, :],
        bb_mask=bb_mask[None, :],
    )


def stack_slots(templates: list[ProteinTemplate]) -> ProteinTemplate:
    """Concatenate a list of ProteinTemplates along the slot axis.

    All inputs must share ``res_num``. Used to combine single-chain template
    slots and complex template slots for the same chain, before the final
    chain-axis ``ProteinTemplate.concat``.
    """
    if not templates:
        msg = "stack_slots needs at least one template."
        raise ValueError(msg)
    n_res = templates[0].res_num
    for t in templates:
        if t.res_num != n_res:
            msg = (
                f"stack_slots: mismatched res_num ({t.res_num} != {n_res}). "
                "All per-chain ProteinTemplates must cover the same residues."
            )
            raise ValueError(msg)
    return ProteinTemplate._from_arrays(  # noqa: SLF001
        mask=np.concatenate([t.mask for t in templates], axis=0),
        ids=np.concatenate([t.ids for t in templates], axis=0),
        res_type=np.concatenate([t.res_type for t in templates], axis=0),
        cb_xyz=np.concatenate([t.cb_xyz for t in templates], axis=0),
        cb_mask=np.concatenate([t.cb_mask for t in templates], axis=0),
        bb_xyz=np.concatenate([t.bb_xyz for t in templates], axis=0),
        bb_mask=np.concatenate([t.bb_mask for t in templates], axis=0),
    )


# ---------------------------------------------------------------------------
# Top-level: build per-chain complex template layers
# ---------------------------------------------------------------------------


def _load_complex_chain(
    complex_spec: "ComplexTemplateSpec",
    template_chain_id: str,
    cif_db: Path | None,
) -> tuple[np.ndarray, list[str]]:
    if complex_spec.cif is not None:
        return _load_chain_backbone_from_cif(complex_spec.cif, template_chain_id)
    if cif_db is None:
        msg = (
            "complex_templates entry uses cif_id but spec.cif_db is not set. "
            "Either provide a cif file path or set spec.cif_db."
        )
        raise ValueError(msg)
    if complex_spec.cif_id is None:  # pragma: no cover — Pydantic validator guards this
        msg = "ComplexTemplateSpec missing both cif and cif_id."
        raise ValueError(msg)
    return _load_chain_backbone_from_lmdb(
        cif_db, complex_spec.cif_id, template_chain_id,
    )


def load_complex_template_layers(
    spec: "InferenceSpec",
    expansions: list["_ChainExpansion"],
    template_id_offset: int = 1000,
) -> list[ProteinTemplate]:
    """Return one ``ProteinTemplate`` per chain whose slot count equals
    ``len(spec.complex_templates)``.

    For each complex template entry, every chain contributes exactly one slot:
    real (mask=True) for chains listed in ``chain_map``, padded empty
    (mask=False) for the rest. This makes slot counts uniform across chains
    so the downstream ``ProteinTemplate.concat`` can stitch residues without
    further padding.
    """
    n_chains = len(expansions)
    if not spec.complex_templates:
        return [ProteinTemplate.empty(exp.n_residues) for exp in expansions]

    per_chain_layers: list[list[ProteinTemplate]] = [[] for _ in range(n_chains)]

    for ki, complex_spec in enumerate(spec.complex_templates):
        template_id = template_id_offset + ki
        loaded: dict[int, tuple[np.ndarray, list[str]]] = {}
        for chain_idx_str, tmpl_chain in complex_spec.chain_map.items():
            try:
                ci = int(chain_idx_str)
            except ValueError as e:
                msg = (
                    f"Complex template[{ki}] chain_map key {chain_idx_str!r} "
                    f"is not a chain index. Use numeric keys (e.g. \"0\", \"1\")."
                )
                raise ValueError(msg) from e
            if not (0 <= ci < n_chains):
                msg = (
                    f"Complex template[{ki}] references chain index {ci}, "
                    f"but only chains 0..{n_chains - 1} exist."
                )
                raise IndexError(msg)
            bb, one_letter = _load_complex_chain(complex_spec, tmpl_chain, spec.cif_db)
            # Align template chain to query — handles "template is full PDB
            # chain, query is a contiguous subsequence" gracefully.
            bb, one_letter = _align_template_to_query(
                bb,
                one_letter,
                expansions[ci].spec.one_letter_seq,
                where=(
                    f"Complex template[{ki}] chain index {ci} "
                    f"(template chain {tmpl_chain!r})"
                ),
            )
            loaded[ci] = (bb, one_letter)

        for ci in range(n_chains):
            if ci in loaded:
                bb, one_letter = loaded[ci]
                per_chain_layers[ci].append(
                    _make_complex_slot_for_chain(bb, one_letter, template_id),
                )
            else:
                per_chain_layers[ci].append(
                    ProteinTemplate.empty(expansions[ci].n_residues),
                )

    return [stack_slots(layers) for layers in per_chain_layers]
