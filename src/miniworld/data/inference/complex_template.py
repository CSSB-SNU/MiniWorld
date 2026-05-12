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
import shutil
import subprocess
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
    """Lay template residues onto the query coordinate columns.

    Fast paths (no external tool):

    * Lengths already match -> accept without checking sequence identity
      (handles modified residues, etc.).
    * Template contains the query as a contiguous substring -> trim to that.

    Fallback: pairwise-align template and query with ``kalign`` and project
    template residues onto query columns. Query-only columns become gaps
    (NaN coords -> bb_mask/cb_mask False downstream); template-only columns
    are dropped (no query residue to place them on).
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
    return _kalign_template_to_query(
        template_bb, template_one_letter, query_one_letter, where=where,
    )


def _kalign_template_to_query(
    template_bb: np.ndarray,
    template_one_letter: list[str],
    query_one_letter: list[str],
    *,
    where: str,
) -> tuple[np.ndarray, list[str]]:
    """kalign-driven pairwise alignment of template to query."""
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
    out_bb = np.full((n_q, *template_bb.shape[1:]), np.nan, dtype=template_bb.dtype)
    out_letters: list[str] = ["-"] * n_q

    qi = 0
    ti = 0
    for q_char, t_char in zip(q_aln, t_aln):
        q_is_res = q_char != "-"
        t_is_res = t_char != "-"
        if q_is_res and t_is_res:
            out_bb[qi] = template_bb[ti]
            out_letters[qi] = template_one_letter[ti]
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
    return out_bb, out_letters


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


def derive_contacts_from_complex_templates(
    spec: "InferenceSpec",
    *,
    positive_cutoff: float = 6.0,
    negative_cutoff: float = 12.0,
    mode: str = "inter",
    seqsep: int = 4,
) -> tuple[list[str], list[str]]:
    """Derive ``(positive, negative)`` contact strings from aligned templates.

    Mirrors the training-time supervision in
    :func:`miniworld.data.features.convert.to_token_contacts`:

    - Per-residue reference coordinate is the **mean of the residue's
      heavy atoms** (read straight from the user-provided cif / cifmol —
      this is independent of MiniWorld's template-model representation,
      which only keeps N/CA/C/CB). This matches training's "average of
      valid atoms in the token" when tokens are residue-level.
    - Pairs with distance ``< positive_cutoff`` (default 6 Å) become
      positive contacts; pairs with ``> negative_cutoff`` (default 12 Å)
      become negative contacts. The gap is intentionally unsupervised.
    - Contacts are emitted at **residue level**
      (``"<chain_letter>:<res_1based>-<chain_letter>:<res_1based>"``).
      :func:`_build_token_contacts` anchors each one at the residue's
      first token, so atomized residues still fold into the token graph.

    Modes:
        - ``inter``: only cross-chain pairs (typical use).
        - ``all``: also includes intra-chain pairs with ``|i - j| >= seqsep``.

    Returns ``([], [])`` when ``spec.complex_templates`` is empty.
    """
    from .fasta import parse_fasta_file

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

    # Per-letter query 1-letter sequences (parsed once across this call).
    letter_to_one_letter: dict[str, list[str]] = {}
    for letter, fasta_path in spec.fasta.items():
        chain_spec = parse_fasta_file(Path(fasta_path), 0)
        letter_to_one_letter[letter] = chain_spec.one_letter_seq

    positive_set: set[tuple[str, int, str, int]] = set()
    negative_set: set[tuple[str, int, str, int]] = set()

    for ki, complex_spec in enumerate(spec.complex_templates):
        centroid_blocks: list[np.ndarray] = []
        chain_letter_per_res: list[str] = []
        local_res_per_pos: list[int] = []

        for chain_idx_str, t_chain_id in complex_spec.chain_map.items():
            try:
                ci = int(chain_idx_str)
            except ValueError as e:
                msg = (
                    f"Complex template[{ki}] chain_map key {chain_idx_str!r} "
                    f"is not a chain index. Use numeric keys (e.g. \"0\", \"1\")."
                )
                raise ValueError(msg) from e
            letter = spec.chain_letters.get(str(ci))
            if letter is None:
                continue
            q_one_letter = letter_to_one_letter.get(letter)
            if q_one_letter is None:
                continue
            t_centroid, t_letters = _load_complex_chain_centroid(
                complex_spec, t_chain_id, spec.cif_db,
            )  # (n_t, 1, 3), one_letter
            aligned_centroid, _aligned_letters = _align_template_to_query(
                t_centroid, t_letters, q_one_letter,
                where=f"complex_templates[{ki}].chain_map['{chain_idx_str}']",
            )
            n_q = len(q_one_letter)
            centroid_blocks.append(aligned_centroid[:, 0, :])  # (n_q, 3)
            chain_letter_per_res.extend([letter] * n_q)
            local_res_per_pos.extend(range(1, n_q + 1))

        if not centroid_blocks:
            continue

        all_c = np.concatenate(centroid_blocks, axis=0)
        chain_arr = np.array(chain_letter_per_res, dtype=object)
        res_arr = np.asarray(local_res_per_pos, dtype=np.int32)
        valid_idx = np.where(np.isfinite(all_c).all(axis=-1))[0]
        if len(valid_idx) < 2:
            continue
        c_valid = all_c[valid_idx]
        diff = c_valid[:, None, :] - c_valid[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        iu, ju = np.triu_indices(len(valid_idx), k=1)
        d = dist[iu, ju]
        pos_mask = d < positive_cutoff
        neg_mask = d > negative_cutoff

        def _accumulate(mask: np.ndarray, target: set) -> None:
            for k in np.where(mask)[0]:
                i, j = int(iu[k]), int(ju[k])
                gi, gj = int(valid_idx[i]), int(valid_idx[j])
                ci, cj = chain_arr[gi], chain_arr[gj]
                ri, rj = int(res_arr[gi]), int(res_arr[gj])
                if mode == "inter" and ci == cj:
                    continue
                if mode == "all" and ci == cj and abs(ri - rj) < seqsep:
                    continue
                if (ci, ri) > (cj, rj):
                    ci, cj, ri, rj = cj, ci, rj, ri
                target.add((ci, ri, cj, rj))

        _accumulate(pos_mask, positive_set)
        _accumulate(neg_mask, negative_set)

    pos_list = [f"{ci}:{ri}-{cj}:{rj}" for (ci, ri, cj, rj) in sorted(positive_set)]
    neg_list = [f"{ci}:{ri}-{cj}:{rj}" for (ci, ri, cj, rj) in sorted(negative_set)]
    return pos_list, neg_list


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
