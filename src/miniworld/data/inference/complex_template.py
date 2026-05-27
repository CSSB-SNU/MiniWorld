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
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from biomol.cif import CIFMol
from biomol.core.types import BioMolDict

from miniworld.data.constants import ResidueMapping
from miniworld.data.io import load_cifmol
from miniworld.data.io.load import load_bytes, load_raw_data
from miniworld.data.pipeline import ProteinTemplate

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .build import _ChainExpansion
    from .spec import ComplexTemplateSpec, InferenceSpec


_BACKBONE_ATOMS = ("N", "CA", "C", "CB")


def _load_assembly_cifmol(
    db_path: Path,
    pdb_id: str,
    assembly_id: str,
    model_id: str,
    alt_id: str,
):
    """Load one ``(assembly, model, alt)`` cifmol from BioMolDB LMDB.

    Auto-detects the two on-disk layouts:

    * **cif.lmdb** (raw structures): outer dict is
      ``{"assembly_dict": {<combo>: ...}, "metadata_dict": ...}``. Returns
      a ``CIFMol`` built via ``CIFMol.from_dict``.
    * **cif_attached_*.lmdb** (training-attached): outer dict has
      ``<combo>`` keys directly, each holding ``"cifmol_attached_dict"``.
      Delegates to :func:`miniworld.data.io.load_cifmol` -> ``CIFMolAttached``.

    Either return type exposes the same ``.chains`` / ``.residues`` /
    ``.atoms`` interface our backbone/centroid/atoms loaders use.
    """
    raw = load_raw_data(pdb_id, db_path)
    if raw is None:
        msg = f"Key '{pdb_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)
    value = load_bytes(raw)
    combo = f"{assembly_id}_{model_id}_{alt_id}"

    if "assembly_dict" in value:
        ad = value["assembly_dict"]
        item = ad.get(combo)
        if item is None:
            available = sorted(ad.keys())
            msg = (
                f"Assembly/model/alt {combo!r} not found for {pdb_id!r}. "
                f"Available: {available}"
            )
            raise KeyError(msg)
        item = BioMolDict(item)
        md = dict(value.get("metadata_dict", {}))
        md["assembly_id"] = assembly_id
        md["model_id"] = model_id
        md["alt_id"] = alt_id
        item["metadata"] = md
        return CIFMol.from_dict(item)

    # cif_attached_*.lmdb path — defer to the canonical loader.
    return load_cifmol(db_path, pdb_id, assembly_id, model_id, alt_id)


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
    cifmol = _load_assembly_cifmol(cif_db_path, pdb_id.lower(), assembly_id, model_id, alt_id)
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
    precomputed: tuple[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Lay template residues onto the query coordinate columns.

    Convenience wrapper around :func:`_query_to_template_index_map`: applies
    the resulting index map to ``template_bb`` and returns the
    query-shaped arrays the existing template-model path expects. Use
    :func:`_query_to_template_index_map` directly if you need the mapping
    for richer per-atom lookups (contact derivation).

    Pass ``precomputed=(query_aln, template_aln)`` to bypass kalign and
    use an HMM-derived alignment (per ``ComplexTemplateSpec.alignment``).
    """
    q_to_t = _query_to_template_index_map(
        template_one_letter, query_one_letter,
        where=where, precomputed=precomputed,
    )
    n_q = len(query_one_letter)
    out_bb = np.full((n_q, *template_bb.shape[1:]), np.nan, dtype=template_bb.dtype)
    out_letters = ["-"] * n_q
    for qi, ti in enumerate(q_to_t):
        if ti < 0:
            continue
        out_bb[qi] = template_bb[ti]
        out_letters[qi] = template_one_letter[ti]
    return out_bb, out_letters


def _query_to_template_index_map(
    template_one_letter: list[str],
    query_one_letter: list[str],
    *,
    where: str = "",
    precomputed: tuple[str, str] | None = None,
) -> np.ndarray:
    """Return an ``(n_q,)`` array mapping query index -> template index (or -1).

    By default (``precomputed=None``) runs kalign pairwise on the query
    and template one-letter sequences. The pairwise path is robust for
    near-identical templates (seq_id ≳ 0.5) but produces garbage
    alignments at twilight-zone identity (≲ 0.3) — see T1331/7zrn
    where kalign matched 84% of residues but only 16% identically,
    putting derive_contacts on the wrong residue pairs entirely.

    Pass ``precomputed=(query_aln, template_aln)`` (two equal-length
    strings with '-' for gaps) to use an HMM-derived alignment instead
    (``scripts/search_template.py --alignment-source hmm`` emits these
    and stores them on :attr:`ComplexTemplateSpec.alignment`).
    """
    if precomputed is not None:
        return _index_map_from_aligned_strings(
            *precomputed,
            n_q=len(query_one_letter),
            n_t=len(template_one_letter),
            where=where,
        )
    return _kalign_index_map(template_one_letter, query_one_letter, where=where)


def _index_map_from_aligned_strings(
    query_aln: str,
    template_aln: str,
    *,
    n_q: int,
    n_t: int,
    where: str,
) -> np.ndarray:
    """Build ``q_to_t`` from two aligned-residue strings of equal length.

    Both strings use uppercase letters for residues and '-' for gaps;
    the two rows must have identical length. Lowercase characters (HMMER
    insertion-state residues) are tolerated on the template row — they
    contribute to the template residue index but not to query alignment.
    """
    if len(query_aln) != len(template_aln):
        msg = (
            f"{where}: precomputed alignment rows have unequal length "
            f"({len(query_aln)} vs {len(template_aln)})"
        )
        raise RuntimeError(msg)
    q_to_t = np.full(n_q, -1, dtype=np.int64)
    qi = 0
    ti = 0
    for q_char, t_char in zip(query_aln, template_aln):
        q_is_match = q_char.isalpha() and q_char.isupper()
        t_is_res = t_char.isalpha()
        if q_is_match and t_is_res:
            if qi < n_q and ti < n_t:
                q_to_t[qi] = ti
            qi += 1
            ti += 1
        elif q_is_match:
            # Query has a match column but template aligns nothing here.
            qi += 1
        elif t_is_res:
            # Template residue not aligned to a query match column
            # (HMMER insert state, lowercase, or query gap).
            ti += 1
    if qi != n_q:
        msg = (
            f"{where}: precomputed alignment ended at query position {qi} "
            f"but query has {n_q} residues — alignment is incomplete."
        )
        raise RuntimeError(msg)
    return q_to_t


def _kalign_index_map(
    template_one_letter: list[str],
    query_one_letter: list[str],
    *,
    where: str,
) -> np.ndarray:
    """Run kalign and return an ``(n_q,)`` query-index -> template-index map.

    Query columns without a template match get ``-1``. Template-only
    columns are silently dropped (no query slot to consume).
    """
    if shutil.which("kalign") is None:
        msg = (
            f"{where}: template length {len(template_one_letter)} != query "
            f"length {len(query_one_letter)} and no substring match. "
            "kalign is required for the fallback alignment but is not on PATH."
        )
        raise RuntimeError(msg)

    query_seq = "".join(query_one_letter)
    template_seq = "".join(template_one_letter)

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.fa"
        out_path = Path(td) / "out.fa"
        in_path.write_text(f">query\n{query_seq}\n>template\n{template_seq}\n")
        try:
            subprocess.run(
                ["kalign", "-i", str(in_path), "-o", str(out_path), "-f", "fasta"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            msg = (
                f"{where}: kalign failed (rc={e.returncode}). "
                f"stderr: {e.stderr.strip()[:500]}"
            )
            raise RuntimeError(msg) from e
        aligned = _read_fasta(out_path)

    if "query" not in aligned or "template" not in aligned:
        msg = f"{where}: kalign output missing query/template records: {list(aligned)}"
        raise RuntimeError(msg)

    q_aln = aligned["query"]
    t_aln = aligned["template"]
    if len(q_aln) != len(t_aln):
        msg = f"{where}: kalign alignment rows have unequal length"
        raise RuntimeError(msg)

    n_q = len(query_one_letter)
    q_to_t = np.full(n_q, -1, dtype=np.int64)
    qi = 0
    ti = 0
    for q_char, t_char in zip(q_aln, t_aln):
        q_is_res = q_char != "-"
        t_is_res = t_char != "-"
        if q_is_res and t_is_res:
            q_to_t[qi] = ti
            qi += 1
            ti += 1
        elif q_is_res:
            qi += 1
        elif t_is_res:
            ti += 1
        # else: both gaps -> skip

    if qi != n_q:
        msg = (
            f"{where}: kalign alignment recovered {qi}/{n_q} query residues; "
            "expected to consume all query positions"
        )
        raise RuntimeError(msg)
    return q_to_t


def _read_fasta(path: Path) -> dict[str, str]:
    """Tiny fasta reader returning {name: sequence} (newlines stripped)."""
    out: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(chunks)
            name = line[1:].split()[0]
            chunks = []
            continue
        chunks.append(line.strip())
    if name is not None:
        out[name] = "".join(chunks)
    return out


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


def _residue_heavy_atom_centroid_biopy(residue) -> np.ndarray:  # noqa: ANN001
    """Mean of heavy-atom xyz for one BioPython residue.

    Hydrogens are dropped via the ``name.startswith("H")`` heuristic, which
    is exact for standard amino acids (only H-prefixed atoms there are
    hydrogens). Returns NaN[3] if the residue has no heavy atoms.
    """
    xyz = [
        atom.get_coord() for atom in residue
        if not atom.get_name().startswith("H")
    ]
    if not xyz:
        return np.full(3, np.nan, dtype=np.float32)
    return np.asarray(xyz, dtype=np.float32).mean(axis=0)


def _load_chain_centroid_from_cif(
    cif_path: Path,
    template_chain_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Per-residue heavy-atom centroid + 1-letter codes from a CIF file.

    Shape ``(n_res, 1, 3)`` so the result plugs into the same
    :func:`_align_template_to_query` machinery used by the backbone path.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True, auth_chains=False)
    parse_path, tmp_path = _open_cif(cif_path)
    try:
        structure = parser.get_structure("complex", parse_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    model = next(iter(structure))
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

    residues = [r for r in target_chain if r.id[0] == " "]
    n_res = len(residues)
    if n_res == 0:
        msg = f"Chain {template_chain_id!r} in {cif_path} has no standard residues."
        raise ValueError(msg)

    centroid = np.full((n_res, 1, 3), np.nan, dtype=np.float32)
    one_letter: list[str] = []
    for ri, res in enumerate(residues):
        centroid[ri, 0] = _residue_heavy_atom_centroid_biopy(res)
        one_letter.append(_three_to_one(res.resname))
    return centroid, one_letter


def _load_chain_centroid_from_lmdb(
    cif_db_path: Path,
    cif_id: str,
    template_chain_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Heavy-atom centroid per residue from a BioMolDB lookup."""
    parts = cif_id.split("_")
    if len(parts) != 4:
        msg = (
            f"cif_id {cif_id!r} must be of the form "
            f"'<pdb_id>_<assembly_id>_<model_id>_<alt_id>'."
        )
        raise ValueError(msg)
    pdb_id, assembly_id, model_id, alt_id = parts
    cifmol = _load_assembly_cifmol(cif_db_path, pdb_id.lower(), assembly_id, model_id, alt_id)
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
    centroid = np.full((n_res, 1, 3), np.nan, dtype=np.float32)
    for ri in range(n_res):
        residue = chain_cifmol.residues[ri]
        atom_ids = np.asarray(residue.atoms.id.value)
        xyz = np.asarray(residue.atoms.xyz.value, dtype=np.float32)
        heavy = np.array(
            [not str(a).startswith("H") for a in atom_ids], dtype=bool,
        )
        if heavy.any():
            centroid[ri, 0] = xyz[heavy].mean(axis=0)
    one_letter_arr = np.asarray(chain_cifmol.residues.one_letter_code_can.value)
    return centroid, [str(c) for c in one_letter_arr]


def _load_complex_chain_centroid(
    complex_spec: "ComplexTemplateSpec",
    template_chain_id: str,
    cif_db: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """Dispatch CIF vs LMDB loaders for the heavy-atom centroid representation."""
    if complex_spec.cif is not None:
        return _load_chain_centroid_from_cif(complex_spec.cif, template_chain_id)
    if cif_db is None:
        msg = (
            "complex_templates entry uses cif_id but spec.cif_db is not set. "
            "Either provide a cif file path or set spec.cif_db."
        )
        raise ValueError(msg)
    if complex_spec.cif_id is None:  # pragma: no cover — guarded by pydantic
        msg = "ComplexTemplateSpec missing both cif and cif_id."
        raise ValueError(msg)
    return _load_chain_centroid_from_lmdb(
        cif_db, complex_spec.cif_id, template_chain_id,
    )


# ---------------------------------------------------------------------------
# Per-residue heavy-atom dict (for token-aware contact derivation)
# ---------------------------------------------------------------------------


def _load_chain_atoms_from_cif(
    cif_path: Path,
    template_chain_id: str,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Per-residue ``{atom_name: xyz}`` (heavy atoms only) + 1-letter codes."""
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True, auth_chains=False)
    parse_path, tmp_path = _open_cif(cif_path)
    try:
        structure = parser.get_structure("complex", parse_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    model = next(iter(structure))
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

    residues = [r for r in target_chain if r.id[0] == " "]
    if not residues:
        msg = f"Chain {template_chain_id!r} in {cif_path} has no standard residues."
        raise ValueError(msg)

    per_residue: list[dict[str, np.ndarray]] = []
    one_letter: list[str] = []
    for res in residues:
        atom_dict: dict[str, np.ndarray] = {}
        for atom in res:
            name = atom.get_name()
            if name.startswith("H"):
                continue
            atom_dict[name] = np.asarray(atom.get_coord(), dtype=np.float32)
        per_residue.append(atom_dict)
        one_letter.append(_three_to_one(res.resname))
    return per_residue, one_letter


def _load_chain_atoms_from_lmdb(
    cif_db_path: Path,
    cif_id: str,
    template_chain_id: str,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Per-residue heavy-atom dict from a BioMolDB lookup."""
    parts = cif_id.split("_")
    if len(parts) != 4:
        msg = (
            f"cif_id {cif_id!r} must be of the form "
            f"'<pdb_id>_<assembly_id>_<model_id>_<alt_id>'."
        )
        raise ValueError(msg)
    pdb_id, assembly_id, model_id, alt_id = parts
    cifmol = _load_assembly_cifmol(cif_db_path, pdb_id.lower(), assembly_id, model_id, alt_id)
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
    per_residue: list[dict[str, np.ndarray]] = []
    for ri in range(n_res):
        residue = chain_cifmol.residues[ri]
        atom_ids = np.asarray(residue.atoms.id.value)
        xyz = np.asarray(residue.atoms.xyz.value, dtype=np.float32)
        atom_dict: dict[str, np.ndarray] = {}
        for a_name, a_xyz in zip(atom_ids, xyz):
            name = str(a_name)
            if name.startswith("H"):
                continue
            atom_dict[name] = a_xyz
        per_residue.append(atom_dict)
    one_letter_arr = np.asarray(chain_cifmol.residues.one_letter_code_can.value)
    return per_residue, [str(c) for c in one_letter_arr]


def _load_complex_chain_atoms(
    complex_spec: "ComplexTemplateSpec",
    template_chain_id: str,
    cif_db: Path | None,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Dispatch CIF vs LMDB loaders for the per-residue atom-dict representation."""
    if complex_spec.cif is not None:
        return _load_chain_atoms_from_cif(complex_spec.cif, template_chain_id)
    if cif_db is None:
        msg = (
            "complex_templates entry uses cif_id but spec.cif_db is not set. "
            "Either provide a cif file path or set spec.cif_db."
        )
        raise ValueError(msg)
    if complex_spec.cif_id is None:  # pragma: no cover — guarded by pydantic
        msg = "ComplexTemplateSpec missing both cif and cif_id."
        raise ValueError(msg)
    return _load_chain_atoms_from_lmdb(
        cif_db, complex_spec.cif_id, template_chain_id,
    )


def _build_derivation_expansions(spec: "InferenceSpec") -> list:
    """Rebuild expansions for standalone contact derivation (skips MSA loading).

    Mirrors the per-chain expansion section of :func:`build_inference_batch`
    but leaves ``msa=None`` since contact derivation never reads it. Used
    when callers don't already have built expansions (e.g. the preview
    CLI). ``build_inference_batch`` itself passes its own expansions in.
    """
    import dataclasses

    from .build import (
        _TERMINAL_ATOM_BY_ENTITY,
        _ChainExpansion,
        _strip_terminal_atoms,
        _tokenize_chain,
    )
    from .ccd import CCDLookup
    from .fasta import parse_fasta_file
    from .tokenization import TokenizationPolicy

    chain_indices = spec.chain_indices()
    ccd_lookup = CCDLookup(spec.ccd_db)
    policy = (
        TokenizationPolicy.from_file(spec.tokenization)
        if spec.tokenization is not None
        else TokenizationPolicy()
    )
    final_chain_letter = {ci: spec.chain_letters[str(ci)] for ci in chain_indices}
    parsed_per_letter: dict[str, object] = {}
    chain_specs: dict[int, object] = {}
    for ci in chain_indices:
        letter = final_chain_letter[ci]
        if letter not in parsed_per_letter:
            parsed_per_letter[letter] = parse_fasta_file(
                spec.fasta[letter], chain_index=ci,
            )
        base = parsed_per_letter[letter]
        chain_specs[ci] = dataclasses.replace(
            base, chain_index=ci, chain_letter=letter,
        )

    expansions: list = []
    atom_offset = 0
    token_offset = 0
    for ci in chain_indices:
        cs = chain_specs[ci]
        residues_full = [ccd_lookup[ccd] for ccd in cs.chemcomp_ids]
        strip_atom = _TERMINAL_ATOM_BY_ENTITY.get(cs.entity_type)
        residues, keep_masks = _strip_terminal_atoms(residues_full, strip_atom)
        n_atoms = sum(r.n_atoms for r in residues)
        atom_to_token_local, token_to_residue_local, residue_token_offsets = (
            _tokenize_chain(cs, residues, residues_full, keep_masks, ccd_lookup, policy)
        )
        n_tokens_chain = int(residue_token_offsets[-1])
        expansions.append(
            _ChainExpansion(
                spec=cs,
                residues=residues,
                n_residues=len(residues),
                n_atoms=n_atoms,
                n_tokens=n_tokens_chain,
                atom_offset=atom_offset,
                token_offset=token_offset,
                msa=None,  # type: ignore[arg-type]  # contact derivation doesn't read msa
                atom_to_token_local=atom_to_token_local,
                token_to_residue_local=token_to_residue_local,
                residue_token_offsets=residue_token_offsets,
            ),
        )
        atom_offset += n_atoms
        token_offset += n_tokens_chain
    return expansions


def derive_contacts_from_complex_templates(
    spec: "InferenceSpec",
    *,
    expansions: list | None = None,
    positive_cutoff: float = 6.0,
    negative_cutoff: float = 12.0,
    mode: str = "all",
    seqsep: int = 4,
) -> tuple[list[str], list[str]]:
    """Derive ``(positive, negative)`` contact strings from aligned templates.

    Mirrors the training-time supervision in
    :func:`miniworld.data.features.convert.to_token_contacts` with proper
    tokenization awareness:

    - For every **query token** (residue-level when ``policy.resolution=1``,
      fragment-level for intermediate, atom-level for 0), the reference
      coordinate is the mean of template atoms whose names match the
      atoms the fragment is composed of. Atoms missing from the template
      (e.g. mutation makes the query-side atom unavailable) are skipped;
      tokens with no matching template atom at all are dropped.
    - Pairs with distance ``< positive_cutoff`` (default 6 Å) become
      positive contacts; pairs with ``> negative_cutoff`` (default 12 Å)
      become negative contacts. The gap is intentionally unsupervised.
    - Contact strings carry the per-residue local token index after ``#``:
      ``"<chain_index>:<res_1based>#<tok_local>-...#..."``.
      :func:`_build_token_contacts` reads the ``#`` and resolves to the
      right global token id. Chain indices (not letters) are used so
      homomeric letters with multiple copies (e.g., ``A2`` mapping
      chains 0 and 3 to letter ``a``) don't collapse distinct
      intra-chain / inter-chain distances into the same key.

    Modes:
        - ``all`` (default): inter-chain pairs plus intra-chain pairs with
          ``|i - j| >= seqsep``. Picks up the template's intra-chain geometry
          too (e.g. a self-prediction supplied as a monomer template), not
          only the cross-chain interface.
        - ``inter``: only cross-chain pairs — useful when the template's
          intra-chain geometry is untrustworthy and you only want to fix
          the interface.

    ``expansions`` lets in-batch callers (``build_inference_batch``) reuse
    their already-built per-chain info. When ``None``, the function
    rebuilds a minimal expansion (no MSA) via :func:`_build_derivation_expansions`.

    Per-chain sequence identity to the query is logged (warning for
    ``< 1.0``, info for exactly 1.0) but never used to gate which chains
    contribute contacts — the per-entry / spec-wide ``as_contact`` flag
    is the only gate. The contact derivation trusts the caller's opt-in.

    Returns ``([], [])`` when ``spec.complex_templates`` is empty.
    """
    if not spec.complex_templates:
        return [], []
    if mode not in ("inter", "all"):
        msg = f"mode must be 'inter' or 'all', got {mode!r}"
        raise ValueError(msg)
    if positive_cutoff >= negative_cutoff:
        msg = (
            f"positive_cutoff ({positive_cutoff}) must be < "
            f"negative_cutoff ({negative_cutoff})"
        )
        raise ValueError(msg)

    # Per-entry opt-in: an entry contributes contacts only when its
    # effective ``as_contact`` is True (per-entry value, or spec-wide
    # default when the entry leaves it None). Entries that opt out stay
    # in the spec — the template module still consumes them as frame
    # templates — they just don't seed contact constraints here.
    contact_entries = [
        (ki, ct) for ki, ct in enumerate(spec.complex_templates)
        if ct.resolves_as_contact(spec.template_as_contact)
    ]
    if not contact_entries:
        return [], []

    if expansions is None:
        expansions = _build_derivation_expansions(spec)
    n_chains = len(expansions)

    # Keys are (chain_idx_i, res_i, tok_i, chain_idx_j, res_j, tok_j). Using
    # chain INDEX (not letter) avoids collapsing distinct intra-chain /
    # inter-chain distances onto the same key when homomeric letters span
    # multiple chains (see docstring).
    positive_set: set[tuple[int, int, int, int, int, int]] = set()
    negative_set: set[tuple[int, int, int, int, int, int]] = set()

    for ki, complex_spec in contact_entries:
        # (chain_idx, res_1based, tok_local, xyz). chain_idx identifies the
        # query chain — distinct entries for distinct chain indices even when
        # they share a letter (homomer copies), so the contact derivation
        # doesn't conflate intra-chain and inter-chain distances.
        per_token: list[tuple[int, int, int, np.ndarray]] = []

        for chain_idx_str, t_chain_id in complex_spec.chain_map.items():
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
            letter = spec.chain_letters.get(str(ci))
            if letter is None:
                continue
            exp = expansions[ci]
            q_one_letter = exp.spec.one_letter_seq

            t_atoms_per_res, t_letters = _load_complex_chain_atoms(
                complex_spec, t_chain_id, spec.cif_db,
            )
            precomputed = None
            if complex_spec.alignment and chain_idx_str in complex_spec.alignment:
                a = complex_spec.alignment[chain_idx_str]
                precomputed = (a["query"], a["template"])
            q_to_t = _query_to_template_index_map(
                t_letters, q_one_letter,
                where=f"complex_templates[{ki}].chain_map['{chain_idx_str}']",
                precomputed=precomputed,
            )

            # Per-chain identity to the query (matches / aligned positions).
            # Same convention casp17/template_plot.py uses for the coverage
            # heatmap so the two stay consistent.
            n_aligned = 0
            n_match = 0
            for qi, ti_idx in enumerate(q_to_t):
                ti_idx_int = int(ti_idx)
                if ti_idx_int < 0:
                    continue
                n_aligned += 1
                if t_letters[ti_idx_int] == q_one_letter[qi]:
                    n_match += 1
            seq_id = (n_match / n_aligned) if n_aligned > 0 else 0.0
            src = complex_spec.cif_id or str(complex_spec.cif)
            tag = (
                f"complex_templates[{ki}] ({src})  query chain {ci} ({letter}) "
                f"<- template chain {t_chain_id}"
            )
            if seq_id < 1.0:
                LOGGER.warning(
                    "%s: seq_id=%.3f < 1.0 -> deriving contacts from a "
                    "non-identical template (as_contact opt-in trusts the "
                    "caller; gate the entry with as_contact=false if undesired)",
                    tag, seq_id,
                )
            else:
                LOGGER.info("%s: seq_id=1.000", tag)

            atom_cursor = 0
            for r_idx in range(exp.n_residues):
                residue = exp.residues[r_idx]
                n_atoms_res = residue.n_atoms
                tok_start = int(exp.residue_token_offsets[r_idx])
                tok_end = int(exp.residue_token_offsets[r_idx + 1])
                n_tok_in_res = tok_end - tok_start
                atom_to_frag = (
                    exp.atom_to_token_local[atom_cursor: atom_cursor + n_atoms_res]
                    - tok_start
                )
                atom_cursor += n_atoms_res

                t_idx = int(q_to_t[r_idx])
                if t_idx < 0:
                    continue  # query residue uncovered by template
                template_atoms = t_atoms_per_res[t_idx]
                if not template_atoms:
                    continue

                for tok_local in range(n_tok_in_res):
                    atom_pos = np.where(atom_to_frag == tok_local)[0]
                    if atom_pos.size == 0:
                        continue
                    atom_names = [str(residue.atom_ids[i]) for i in atom_pos]
                    xyz = [
                        template_atoms[name] for name in atom_names
                        if name in template_atoms
                    ]
                    if not xyz:
                        continue
                    token_xyz = np.mean(np.asarray(xyz, dtype=np.float32), axis=0)
                    per_token.append((ci, r_idx + 1, tok_local, token_xyz))

        if len(per_token) < 2:
            continue

        coords = np.stack([t[3] for t in per_token], axis=0)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        iu, ju = np.triu_indices(len(per_token), k=1)
        d = dist[iu, ju]
        pos_mask = d < positive_cutoff
        neg_mask = d > negative_cutoff

        def _accumulate(mask: np.ndarray, target: set) -> None:
            for k in np.where(mask)[0]:
                i, j = int(iu[k]), int(ju[k])
                ci_a, ri, ti_tok, _ = per_token[i]
                ci_b, rj, tj_tok, _ = per_token[j]
                # Same-chain filter is now by chain INDEX, so homomer copies
                # (distinct chain indices sharing a letter) are correctly
                # treated as inter-chain.
                if mode == "inter" and ci_a == ci_b:
                    continue
                if mode == "all" and ci_a == ci_b and abs(ri - rj) < seqsep:
                    continue
                if (ci_a, ri, ti_tok) > (ci_b, rj, tj_tok):
                    ci_a, ci_b = ci_b, ci_a
                    ri, rj = rj, ri
                    ti_tok, tj_tok = tj_tok, ti_tok
                target.add((ci_a, ri, ti_tok, ci_b, rj, tj_tok))

        _accumulate(pos_mask, positive_set)
        _accumulate(neg_mask, negative_set)

    def _fmt(k: tuple) -> str:
        ci_a, ri, ti, ci_b, rj, tj = k
        # Numeric chain indices; _build_token_contacts resolves these without
        # the letter->[chain_indices] Cartesian expansion that previously
        # broadcast intra-chain contacts onto inter-chain homomer pairs.
        return f"{ci_a}:{ri}#{ti}-{ci_b}:{rj}#{tj}"

    pos_list = [_fmt(k) for k in sorted(positive_set)]
    neg_list = [_fmt(k) for k in sorted(negative_set)]
    return pos_list, neg_list


def load_complex_template_layers(
    spec: "InferenceSpec",
    expansions: list["_ChainExpansion"],
    template_id_offset: int = 1000,
) -> list[ProteinTemplate]:
    """Return one ``ProteinTemplate`` per chain whose slot count equals
    the total number of (template × covered-condition-group) emissions.

    For each complex template entry, *every condition-group it covers
    emits one independent slot*: real (mask=True) for chains listed in
    ``chain_map`` AND belonging to that group, padded empty (mask=False)
    for the rest. This is the template-side counterpart of the MSA
    pairing's group-aware row blocks — a single multi-chain template that
    straddles the antibody / antigen boundary (e.g. a 4-chain Ab-Ag co-
    crystal `7lbg`) gets split into "intra-Ab geometry" and "intra-Ag
    geometry" slots, so the diffusion is free to choose the inter-group
    relative pose without template-imposed conditioning.

    When ``spec.condition_groups`` is empty, the behaviour collapses to
    the pre-split scheme: one slot per template entry, real for covered
    chains, empty for the rest. With groups declared, chains absent from
    every declared group become their own singleton group (so a template
    covering only that chain still emits exactly one slot, geometry intact).
    """
    n_chains = len(expansions)
    if not spec.complex_templates:
        return [ProteinTemplate.empty(exp.n_residues) for exp in expansions]

    # Resolve chain → group_id. Empty ``condition_groups`` means *no
    # grouping*: assign every chain to a single shared group so each
    # template emits one combined slot (legacy behaviour). When groups
    # are declared, chains outside any group become their own singleton
    # (group_id < 0 namespace).
    chain_to_group: dict[int, int] = {}
    if spec.condition_groups:
        for gi, group in enumerate(spec.condition_groups):
            for ci in group:
                chain_to_group[int(ci)] = gi
        next_singleton = -1
        for ci in range(n_chains):
            if ci not in chain_to_group:
                chain_to_group[ci] = next_singleton
                next_singleton -= 1
    else:
        for ci in range(n_chains):
            chain_to_group[ci] = 0

    per_chain_layers: list[list[ProteinTemplate]] = [[] for _ in range(n_chains)]
    slot_counter = template_id_offset

    for ki, complex_spec in enumerate(spec.complex_templates):
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
            # chain, query is a contiguous subsequence" gracefully. HMM-
            # derived alignment (if stored on the spec) bypasses kalign.
            precomputed = None
            if complex_spec.alignment and chain_idx_str in complex_spec.alignment:
                a = complex_spec.alignment[chain_idx_str]
                precomputed = (a["query"], a["template"])
            bb, one_letter = _align_template_to_query(
                bb,
                one_letter,
                expansions[ci].spec.one_letter_seq,
                where=(
                    f"Complex template[{ki}] chain index {ci} "
                    f"(template chain {tmpl_chain!r})"
                ),
                precomputed=precomputed,
            )
            loaded[ci] = (bb, one_letter)

        # Group the loaded chains by their condition_group id. Preserve
        # encounter order so the slot index is deterministic.
        groups_in_template: list[int] = []
        chains_per_group: dict[int, list[int]] = {}
        for ci in loaded:
            gid = chain_to_group[ci]
            if gid not in chains_per_group:
                groups_in_template.append(gid)
                chains_per_group[gid] = []
            chains_per_group[gid].append(ci)

        # Emit one slot per condition-group that the template covers.
        for gid in groups_in_template:
            slot_id = slot_counter
            slot_counter += 1
            keep = set(chains_per_group[gid])
            for ci in range(n_chains):
                if ci in keep:
                    bb, one_letter = loaded[ci]
                    per_chain_layers[ci].append(
                        _make_complex_slot_for_chain(bb, one_letter, slot_id),
                    )
                else:
                    per_chain_layers[ci].append(
                        ProteinTemplate.empty(expansions[ci].n_residues),
                    )

    return [stack_slots(layers) for layers in per_chain_layers]
