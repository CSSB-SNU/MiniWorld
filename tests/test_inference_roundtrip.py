"""Roundtrip test: reverse a dataloader Batch into fasta/a3m/contacts/spec.yaml.

The dataloader builds a ``Batch`` from LMDB-backed CIF + a3m + template +
contacts. This script reverses that Batch into the YAML spec consumed by the
new inference path (``miniworld.data.inference``), so the same sample can be
fed back through ``build_inference_batch`` for end-to-end equivalence checks.

Two subcommands:
  * ``dump`` — write fasta + a3m + spec.yaml for one dataset item.
  * ``roundtrip`` — ``dump`` plus ``build_inference_batch`` and per-tensor
    diff against the original Batch.

Example:
    python tests/test_inference_roundtrip.py dump --name 7PTQ_1_1_. \\
        --out-dir tests/artifacts/roundtrip/7PTQ
    python tests/test_inference_roundtrip.py roundtrip --name 7PTQ_1_1_. \\
        --out-dir tests/artifacts/roundtrip/7PTQ
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import torch
import yaml

from miniworld.data.constants import ResidueMapping
from miniworld.data.features import Batch
from miniworld.data.inference import build_inference_batch
from miniworld.data.inference.fasta import (
    EntityType,
    _auto_detect_entity_type,
    index_to_lowercase_letter,
)
from miniworld.data.inference.spec import ContactsSpec, InferenceSpec
from miniworld.data.io import load_cifmol
from miniworld.data.io.load import load_a3m  # not re-exported via package __init__

# Reuse the existing dataloader CLI helpers from test_dataloader.py.
# When run as a script, ``tests/`` is on sys.path so a sibling import works;
# when imported as ``tests.test_inference_roundtrip``, fall back to the package.
try:
    from test_dataloader import (  # type: ignore[import-not-found]
        ItemOptions,
        build_dataset,
        dataset_options,
        ensure_paths_exist,
        fetch_direct_item,
        parse_sample_name,
    )
except ModuleNotFoundError:  # pragma: no cover
    from tests.test_dataloader import (
        ItemOptions,
        build_dataset,
        dataset_options,
        ensure_paths_exist,
        fetch_direct_item,
        parse_sample_name,
    )


# Inference test fixtures must preserve the entire assembly (no spatial crop),
# so we always run the dataloader with effectively-infinite token/atom budgets.
# ``crop_spatial_segment_token`` then visits every residue in distance order
# until exhausted, yielding the full cifmol unchanged.
_NO_CROP_TOKENS = 10_000_000
_NO_CROP_ATOMS = 10_000_000


def _disable_cropping(args: ItemOptions) -> ItemOptions:
    return dataclasses.replace(
        args,
        max_tokens=_NO_CROP_TOKENS,
        max_atoms=_NO_CROP_ATOMS,
    )


# ---------------------------------------------------------------------------
# Inverse residue / entity mappings
# ---------------------------------------------------------------------------


def _aa3_to_aa1() -> dict[str, str]:
    return {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }


def _dna_ccd_to_one() -> dict[str, str]:
    return {"DA": "A", "DT": "T", "DG": "G", "DC": "C"}


_RM = ResidueMapping()
_AA_NUM_TO_LETTER = list(_RM.NUM2AA)  # index by token_type int; "X" for unknown
_RNA_NUM_TO_LETTER = {21: "A", 22: "U", 23: "G", 24: "C", 25: "X"}
_DNA_NUM_TO_LETTER = {26: "A", 27: "T", 28: "G", 29: "C", 30: "X"}
_GAP_INDEX = _RM.protein.GAP_INDEX  # 31


_ENTITY_INT_TO_FASTA: dict[int, str] = {
    0: "polypeptide(L)",  # antibody → treat as polypeptide
    1: "polypeptide(L)",
    2: "polypeptide(L)",  # d-protein
    3: "polyribonucleotide",
    4: "polydeoxyribonucleotide",
    5: "polyribonucleotide",  # NA → fall back to RNA
    6: "non-polymer",
    7: "branched",
}


def _decode_msa_letter(token_int: int, entity_int: int) -> str:
    """Decode an MSA integer code back to its a3m letter (uppercase or '-')."""
    if token_int == _GAP_INDEX:
        return "-"
    if entity_int in (3, 5):  # RNA / NA
        return _RNA_NUM_TO_LETTER.get(token_int, "X")
    if entity_int == 4:  # DNA
        return _DNA_NUM_TO_LETTER.get(token_int, "X")
    if entity_int in (6, 7):  # ligand / branched
        return "X"
    return _AA_NUM_TO_LETTER[token_int] if 0 <= token_int < len(_AA_NUM_TO_LETTER) else "X"


# ---------------------------------------------------------------------------
# Chain-level extraction from a Batch
# ---------------------------------------------------------------------------


@dataclass
class ChainSlice:
    """Per-chain info extracted from a Batch (residue-level tokenization assumed)."""

    chain_idx: int
    chain_letter: str
    entity_int: int
    entity_fasta: str
    token_lo: int
    token_hi: int
    chemcomp_ids: list[str]


def chain_index_to_letter(i: int) -> str:
    """Synthesize a unique chain letter from a chain index (A..Z, AA..ZZ, ...)."""
    if i < 0:
        msg = f"chain index must be non-negative, got {i}."
        raise ValueError(msg)
    out = []
    n = i
    while True:
        out.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(out))


def extract_chain_slices(batch: Batch) -> list[ChainSlice]:
    """Build per-chain slices from a B=1 dataloader Batch."""
    if batch.chain.entity_type.shape[0] != 1:
        msg = "Roundtrip only supports B=1 batches."
        raise ValueError(msg)

    entity_arr = batch.chain.entity_type[0].cpu().numpy().astype(np.int64)
    token_asym = batch.scheme.token_asym_id[0].cpu().numpy().astype(np.int64)
    # ``batch.chem_comp_ids[0]`` may be a BioMol NodeFeature (dataloader path)
    # or a plain object array (inference path); unwrap to a flat string array.
    raw_ccd = batch.chem_comp_ids[0]
    if hasattr(raw_ccd, "value"):
        raw_ccd = raw_ccd.value
    chem_comp = [str(x) for x in np.asarray(raw_ccd)]

    n_chains = int(entity_arr.shape[0])
    slices: list[ChainSlice] = []
    for ci in range(n_chains):
        token_idx_in_chain = np.where(token_asym == ci)[0]
        if token_idx_in_chain.size == 0:
            msg = (
                f"Chain {ci} has no tokens — cannot reconstruct a fasta entry. "
                f"This batch likely has an unusual scheme; skip or filter."
            )
            raise ValueError(msg)
        # Token order is increasing within each chain (chains are contiguous).
        token_lo = int(token_idx_in_chain.min())
        token_hi = int(token_idx_in_chain.max()) + 1
        if token_hi - token_lo != token_idx_in_chain.size:
            msg = f"Chain {ci} tokens are not contiguous: {token_idx_in_chain.tolist()}"
            raise ValueError(msg)
        chemcomp_ids = [str(chem_comp[t]) for t in range(token_lo, token_hi)]
        ent = int(entity_arr[ci])
        slices.append(
            ChainSlice(
                chain_idx=ci,
                chain_letter=chain_index_to_letter(ci),
                entity_int=ent,
                entity_fasta=_ENTITY_INT_TO_FASTA.get(ent, "non-polymer"),
                token_lo=token_lo,
                token_hi=token_hi,
                chemcomp_ids=chemcomp_ids,
            ),
        )
    return slices


# ---------------------------------------------------------------------------
# Reverse: Batch -> fasta body
# ---------------------------------------------------------------------------


def _polymer_one_letter(chain: ChainSlice, batch: Batch) -> str:
    """1-letter sequence for a polymer chain.

    Uses MSA query (row 0) so we round-trip exactly the integers the inference
    path will see; falls back to 3-letter→1-letter CCD lookup when the MSA row
    is a gap (shouldn't happen for query but guards nonetheless).
    """
    aa_lookup = _aa3_to_aa1()
    dna_lookup = _dna_ccd_to_one()

    query = batch.msa.aligned_sequences[0, 0, chain.token_lo: chain.token_hi].cpu().numpy()
    out = []
    for pos, ccd in enumerate(chain.chemcomp_ids):
        token_int = int(query[pos])
        letter = _decode_msa_letter(token_int, chain.entity_int)
        if letter in {"-", "X"}:
            # Fall back to canonical CCD-based 1-letter.
            if chain.entity_int in (3, 5):
                letter = ccd if len(ccd) == 1 else "X"
            elif chain.entity_int == 4:
                letter = dna_lookup.get(ccd, "X")
            else:
                letter = aa_lookup.get(ccd, "X")
        out.append(letter)
    return "".join(out)


def _branched_body(chain: ChainSlice, batch: Batch) -> str:
    """Branched body ``(CCD1)(CCD2)...|(i,j)...`` with 1-based bond indices."""
    ccds = chain.chemcomp_ids
    body = "".join(f"({c})" for c in ccds)
    token_bond = batch.structure.token_bond[0].cpu().numpy()
    pairs = []
    for a, b in token_bond:
        a, b = int(a), int(b)
        if chain.token_lo <= a < chain.token_hi and chain.token_lo <= b < chain.token_hi:
            la = a - chain.token_lo + 1
            lb = b - chain.token_lo + 1
            if la == lb:
                continue
            pairs.append((min(la, lb), max(la, lb)))
    if pairs:
        body += "|" + "".join(f"({i},{j})" for i, j in pairs)
    return body


def chain_to_fasta(chain: ChainSlice, batch: Batch, sample_name: str) -> str:
    """Render one chain's fasta record (header + body)."""
    header = f">{sample_name}_{chain.chain_letter} | {chain.entity_fasta} | Chain:{chain.chain_letter}"
    if chain.entity_fasta == "non-polymer":
        # Always exactly one CCD per non-polymer chain.
        body = f"({chain.chemcomp_ids[0]})"
    elif chain.entity_fasta == "branched":
        body = _branched_body(chain, batch)
    else:
        body = _polymer_one_letter(chain, batch)
    return f"{header}\n{body}\n"


# ---------------------------------------------------------------------------
# Reverse: Batch -> a3m text per chain
# ---------------------------------------------------------------------------


def _decode_deletion_count(deletion_value: float) -> int:
    """Inverse of ComplexMSA's ``2*arctan(count/3)/π`` mapping."""
    if deletion_value <= 0.0:
        return 0
    if deletion_value >= 1.0:
        # Saturation: tan(π/2) is infinite; clamp at a generous bound.
        return 99
    return max(0, int(round(3.0 * math.tan(0.5 * math.pi * deletion_value))))


def chain_to_a3m(chain: ChainSlice, batch: Batch) -> str:
    """Fallback: render a3m from the (already-sampled) Batch MSA rows.

    Used when the LMDB seq_id is unknown. Species info is **lost** because
    ``ComplexMSA`` drops it during pair-and-sample, so headers are minimal.
    """
    n_msa = batch.msa.aligned_sequences.shape[1]
    aligned = batch.msa.aligned_sequences[0, :, chain.token_lo: chain.token_hi].cpu().numpy()
    has_del = batch.msa.has_deletion[0, :, chain.token_lo: chain.token_hi].cpu().numpy()
    del_val = batch.msa.deletion_value[0, :, chain.token_lo: chain.token_hi].cpu().numpy()
    msa_mask = batch.msa.mask[0].cpu().numpy()

    lines = []
    for k in range(n_msa):
        if msa_mask[k] == 0:
            continue
        chars: list[str] = []
        L = chain.token_hi - chain.token_lo
        for pos in range(L):
            if has_del[k, pos]:
                count = _decode_deletion_count(float(del_val[k, pos]))
                if count > 0:
                    chars.append("x" * count)
            chars.append(_decode_msa_letter(int(aligned[k, pos]), chain.entity_int))
        header = ">query" if k == 0 else f">seq{k}"
        lines.append(header)
        lines.append("".join(chars))
    return "\n".join(lines) + "\n"


def chain_to_a3m_lmdb(
    seq_id: str,
    a3m_db_path: Path,
    entity_int: int,
    expected_len: int | None = None,
) -> str | None:
    """Render full a3m text directly from the LMDB-stored MSA, with species headers.

    Preserves the entire alignment (not the dataloader's sampled subset) and the
    ``species`` array stored alongside ``aligned_sequences`` / ``deletions``.
    Returns ``None`` if no MSA exists for ``seq_id`` (caller falls back), or if
    ``expected_len`` is provided and the raw MSA length disagrees (crops can't
    be reproduced without ``crop_indices``).
    """
    msa = load_a3m(seq_id, a3m_db_path)
    if msa is None:
        return None
    aligned = msa.aligned_sequences  # (N, L) int
    deletions = msa.deletions        # (N, L) uint8
    species = msa.species            # (N,) object/str

    n_seqs, L = aligned.shape
    if expected_len is not None and L != expected_len:
        return None
    lines: list[str] = []
    for k in range(n_seqs):
        chars: list[str] = []
        for pos in range(L):
            count = int(deletions[k, pos])
            if count > 0:
                # Original insertion letters are not preserved in the LMDB
                # (only counts), so emit a stable lowercase placeholder.
                chars.append("x" * count)
            chars.append(_decode_msa_letter(int(aligned[k, pos]), entity_int))
        sp_raw = species[k]
        if isinstance(sp_raw, bytes):
            sp_raw = sp_raw.decode("utf-8", errors="replace")
        sp = str(sp_raw) if sp_raw not in (None, "", "N/A") else "N/A"
        if k == 0:
            # Query header: minimal, parser treats row 0 as 'query' regardless.
            header = f">query|{seq_id}|query OS={sp} GN=N/A"
        else:
            # UniProt-pipe layout matches StructCooker's parse_headers Pattern 2,
            # so the species field round-trips through ``parse_a3m_file``.
            db_id = f"{seq_id}_{k}"
            header = (
                f">tr|{db_id}|{db_id}_LMDB sequence "
                f"OS={sp} GN=N/A PE=3 SV=1"
            )
        lines.append(header)
        lines.append("".join(chars))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reverse: Batch -> contacts JSON
# ---------------------------------------------------------------------------


def _global_token_to_chain_residx(
    token: int,
    chains: list[ChainSlice],
) -> tuple[str, int]:
    """Return ``(chain_letter, 1-based residue idx within chain)`` for a global token."""
    for c in chains:
        if c.token_lo <= token < c.token_hi:
            return c.chain_letter, token - c.token_lo + 1
    msg = f"Token index {token} is outside any chain range."
    raise ValueError(msg)


def _pair_to_contact_string(i: int, j: int, chains: list[ChainSlice]) -> str:
    ca, ra = _global_token_to_chain_residx(i, chains)
    cb, rb = _global_token_to_chain_residx(j, chains)
    return f"{ca}:{ra}-{cb}:{rb}"


def compute_dense_contacts(
    batch: Batch,
    chains: list[ChainSlice],
    positive_cutoff: float = 6.0,
    negative_cutoff: float = 12.0,
    max_positive: int | None = None,
    max_negative: int | None = None,
    seed: int = 0,
) -> ContactsSpec:
    """All token-token pairs from GT atom positions (no Poisson sub-sampling).

    Token positions = mean of valid atoms per token (matching the pooling rule
    in ``miniworld.data.features.convert.to_token_contacts``). A pair is
    classified as ``positive`` when its distance is below ``positive_cutoff``,
    ``negative`` when above ``negative_cutoff``, and dropped otherwise.
    Optionally caps each side via random sub-sampling.
    """
    rng = np.random.default_rng(seed)

    atom_pos = batch.structure.atom_pos[0].cpu().numpy().astype(np.float64)
    atom_pos_mask = batch.structure.atom_pos_mask[0].cpu().numpy().astype(bool)
    atom_to_token = batch.scheme.atom_to_token_idx_map[0].cpu().numpy().astype(np.int64)
    n_tokens = int(batch.token_length)

    valid_atom = atom_pos_mask & np.isfinite(atom_pos).all(axis=1)
    token_sum = np.zeros((n_tokens, 3), dtype=np.float64)
    token_count = np.zeros(n_tokens, dtype=np.float64)
    np.add.at(token_sum, atom_to_token[valid_atom], atom_pos[valid_atom])
    np.add.at(token_count, atom_to_token[valid_atom], 1.0)
    valid_token = token_count > 0
    token_pos = np.where(
        valid_token[:, None],
        token_sum / np.maximum(token_count, 1.0)[:, None],
        0.0,
    )

    iu, ju = np.triu_indices(n_tokens, k=1)
    valid_pair = valid_token[iu] & valid_token[ju]
    diff = token_pos[iu] - token_pos[ju]
    dists = np.sqrt(np.einsum("nd,nd->n", diff, diff))

    pos_sel = valid_pair & (dists < positive_cutoff)
    neg_sel = valid_pair & (dists > negative_cutoff)

    pos_pairs = list(zip(iu[pos_sel].tolist(), ju[pos_sel].tolist()))
    neg_pairs = list(zip(iu[neg_sel].tolist(), ju[neg_sel].tolist()))

    if max_positive is not None and len(pos_pairs) > max_positive:
        idx = rng.choice(len(pos_pairs), max_positive, replace=False)
        pos_pairs = [pos_pairs[k] for k in idx]
    if max_negative is not None and len(neg_pairs) > max_negative:
        idx = rng.choice(len(neg_pairs), max_negative, replace=False)
        neg_pairs = [neg_pairs[k] for k in idx]

    positive = [_pair_to_contact_string(i, j, chains) for i, j in pos_pairs]
    negative = [_pair_to_contact_string(i, j, chains) for i, j in neg_pairs]
    return ContactsSpec(positive=positive, negative=negative)


def batch_to_contacts(batch: Batch, chains: list[ChainSlice]) -> ContactsSpec:
    """Decode ``batch.structure.token_contacts`` into the spec's "A:5-B:12" form."""
    pairs = batch.structure.token_contacts[0].cpu().numpy()
    positive: list[str] = []
    negative: list[str] = []
    for ti, tj, t in pairs:
        ti, tj, t = int(ti), int(tj), int(t)
        if ti == 0 and tj == 0 and t == 0 and pairs.shape[0] == 1:
            # Empty placeholder used by some pipelines; skip.
            continue
        ca, ra = _global_token_to_chain_residx(ti, chains)
        cb, rb = _global_token_to_chain_residx(tj, chains)
        s = f"{ca}:{ra}-{cb}:{rb}"
        (positive if t == 0 else negative).append(s)
    return ContactsSpec(positive=positive, negative=negative)


# ---------------------------------------------------------------------------
# Top-level dump
# ---------------------------------------------------------------------------


def _resolve_chain_seq_ids(
    batch: Batch,
    cif_db_path: Path | None,
    chains: list[ChainSlice],
) -> list[str | None]:
    """Look up the cifmol ``seq_id`` for each Batch chain (for raw-MSA loading)."""
    if cif_db_path is None:
        return [None] * len(chains)
    try:
        pdb_id, assembly_id, model_id, alt_id = parse_sample_name(str(batch.name[0]))
    except Exception:
        return [None] * len(chains)
    try:
        cifmol = load_cifmol(cif_db_path, pdb_id, assembly_id, model_id, alt_id)
    except Exception:
        return [None] * len(chains)
    raw = cifmol.chains.seq_id.value
    arr = list(np.asarray(raw))
    out: list[str | None] = []
    for c in chains:
        out.append(str(arr[c.chain_idx]) if c.chain_idx < len(arr) else None)
    return out


def dump_inference_spec(
    batch: Batch,
    out_dir: Path,
    ccd_db_path: Path,
    cif_db_path: Path | None = None,
    a3m_db_path: Path | None = None,
    template_db_path: Path | None = None,
    template_n: int = 4,
    contacts_override: ContactsSpec | None = None,
    spec_name_override: str | None = None,
) -> tuple[InferenceSpec, list[ChainSlice]]:
    """Write fasta + a3m files and a spec.yaml into ``out_dir``.

    ``cif_db_path`` + ``a3m_db_path`` are optional but recommended: when both
    are set, this loads the **raw** MSA from the a3m LMDB (preserving species
    headers and the full alignment depth), instead of the dataloader's
    sub-sampled, species-stripped per-Batch MSA.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_name = str(batch.name[0])

    chains = extract_chain_slices(batch)
    seq_ids = _resolve_chain_seq_ids(batch, cif_db_path, chains)
    chain_letters_map: dict[str, str] = {str(c.chain_idx): c.chain_letter for c in chains}
    # fasta / a3m are letter-keyed; dataloader-extracted chains all have
    # distinct letters (chain_index_to_letter is unique per index), so
    # the loops below write one file per letter without collisions.
    fasta_paths: dict[str, str] = {}
    a3m_paths: dict[str, str] = {}
    for c, seq_id in zip(chains, seq_ids):
        fasta_path = out_dir / f"chain_{c.chain_idx}_{c.chain_letter}.fasta"
        fasta_path.write_text(chain_to_fasta(c, batch, sample_name))
        fasta_paths[c.chain_letter] = str(fasta_path)

        # Only emit a3m for polymers (RNA/DNA/protein); ligand/branched a3m is
        # built on-the-fly via ``MSA.from_query`` in the inference path.
        if c.entity_fasta in ("polypeptide(L)", "polyribonucleotide", "polydeoxyribonucleotide"):
            a3m_text: str | None = None
            if seq_id is not None and a3m_db_path is not None:
                expected_len = c.token_hi - c.token_lo  # residue-level tokenization
                a3m_text = chain_to_a3m_lmdb(
                    seq_id, a3m_db_path, c.entity_int, expected_len=expected_len,
                )
            if a3m_text is None:
                a3m_text = chain_to_a3m(c, batch)
            a3m_path = out_dir / f"chain_{c.chain_idx}_{c.chain_letter}.a3m"
            a3m_path.write_text(a3m_text)
            a3m_paths[c.chain_letter] = str(a3m_path)

    contacts = contacts_override if contacts_override is not None else batch_to_contacts(batch, chains)

    # Templates: when ``template_db_path`` is set, populate spec.template with
    # each polymer chain's seq_id (the LMDB key in the StructCooker template
    # DB). Non-polymer / branched chains are left out of the dict so the
    # inference build path emits empty slots for them.
    template_map: dict[str, str] = {}
    if template_db_path is not None:
        for c, seq_id in zip(chains, seq_ids):
            if seq_id is None:
                continue
            if c.entity_fasta not in (
                "polypeptide(L)", "polyribonucleotide", "polydeoxyribonucleotide",
            ):
                continue
            template_map[str(c.chain_idx)] = seq_id

    spec_name = spec_name_override or f"roundtrip_{sample_name}"
    spec = InferenceSpec(
        name=spec_name,
        chain_letters=chain_letters_map,
        fasta={k: Path(v) for k, v in fasta_paths.items()},
        ccd_db=ccd_db_path,
        a3m={k: Path(v) for k, v in a3m_paths.items()},
        template_db=template_db_path,
        template=template_map,
        template_n=template_n,
        contacts=contacts,
    )

    spec_dict: dict[str, Any] = {
        "name": spec.name,
        "chain_letters": chain_letters_map,
        "fasta": fasta_paths,
        "ccd_db": str(ccd_db_path),
        "a3m": a3m_paths,
        "contacts": {"positive": contacts.positive, "negative": contacts.negative},
    }
    if template_db_path is not None:
        spec_dict["template_db"] = str(template_db_path)
        spec_dict["template"] = template_map
        spec_dict["template_n"] = template_n
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return spec, chains


# ---------------------------------------------------------------------------
# Roundtrip equivalence check
# ---------------------------------------------------------------------------


@dataclass
class FieldDiff:
    name: str
    ok: bool
    expected_diff: bool  # True when divergence is structural (canonical CCD vs cifmol)
    detail: str


def _eq_tensor(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-5) -> bool:
    if a.shape != b.shape:
        return False
    if a.is_floating_point() or b.is_floating_point():
        return torch.allclose(a.float(), b.float(), atol=atol)
    return torch.equal(a, b)


def compare_batches(orig: Batch, rebuilt: Batch) -> list[FieldDiff]:
    """Compare Batch fields; flags fields whose divergence is by-design.

    Fields marked ``expected_diff`` differ for structural reasons:
      * ``scheme.token_residue_idx`` — orig uses CIF residue numbering,
        rebuilt uses 0..N (no CIF metadata in the spec).
      * ``scheme.token_entity_id`` — orig uses cifmol entity ids, rebuilt
        groups by (entity_type, ccd-tuple); offset is benign.
      * Atom-level fields (``atom_mask``, ``atom_to_token_idx_map``,
        ``atom_to_chain_id``, ``reference.*``) — orig uses the realised
        cifmol atom set (with terminal OXT removed, missing atoms etc.),
        rebuilt uses canonical CCD topology.
    """

    diffs: list[FieldDiff] = []

    def add(name: str, ok: bool, *, expected_diff: bool = False, detail: str = "") -> None:
        diffs.append(FieldDiff(name=name, ok=ok, expected_diff=expected_diff, detail=detail))

    n_atoms_orig = int(orig.structure.atom_mask.shape[1])
    n_atoms_new = int(rebuilt.structure.atom_mask.shape[1])
    atom_count_diverges = n_atoms_orig != n_atoms_new

    # Scheme — token-level fields are the ground truth for "did topology survive".
    for k in ("token_idx", "token_asym_id"):
        a = getattr(orig.scheme, k).cpu()
        b = getattr(rebuilt.scheme, k).cpu()
        ok = _eq_tensor(a, b)
        add(f"scheme.{k}", ok, detail="" if ok else f"orig={a.tolist()[:8]}... rebuilt={b.tolist()[:8]}...")

    for k in ("token_residue_idx", "token_entity_id"):
        a = getattr(orig.scheme, k).cpu()
        b = getattr(rebuilt.scheme, k).cpu()
        ok = _eq_tensor(a, b)
        add(f"scheme.{k}", ok, expected_diff=True, detail="" if ok else "convention/offset difference (informational)")

    for k in ("atom_to_token_idx_map", "atom_to_chain_id"):
        a = getattr(orig.scheme, k).cpu()
        b = getattr(rebuilt.scheme, k).cpu()
        ok = _eq_tensor(a, b)
        add(
            f"scheme.{k}",
            ok,
            expected_diff=atom_count_diverges,
            detail="" if ok else f"shape orig={tuple(a.shape)} vs rebuilt={tuple(b.shape)}",
        )

    add(
        "chain.entity_type",
        _eq_tensor(orig.chain.entity_type.cpu(), rebuilt.chain.entity_type.cpu()),
    )
    add(
        "sequence.token_type",
        _eq_tensor(orig.sequence.token_type.cpu(), rebuilt.sequence.token_type.cpu()),
    )
    add(
        "structure.atom_mask",
        _eq_tensor(orig.structure.atom_mask.cpu(), rebuilt.structure.atom_mask.cpu()),
        expected_diff=atom_count_diverges,
    )
    add(
        "structure.token_mask",
        _eq_tensor(orig.structure.token_mask.cpu(), rebuilt.structure.token_mask.cpu()),
    )

    # token_bond: order doesn't matter; compare as a set of pairs.
    a_pairs = {tuple(p.tolist()) for p in orig.structure.token_bond[0].cpu()}
    b_pairs = {tuple(p.tolist()) for p in rebuilt.structure.token_bond[0].cpu()}
    # Both sides share the [[0,0]] placeholder when there are no real edges; ignore it.
    placeholder = (0, 0)
    a_real = a_pairs - {placeholder}
    b_real = b_pairs - {placeholder}
    add(
        "structure.token_bond (set, ignoring [0,0] placeholder)",
        a_real == b_real,
        detail="" if a_real == b_real else f"only_in_orig={a_real - b_real}, only_in_rebuilt={b_real - a_real}",
    )

    # token_contacts: same set after canonicalizing (a < b, type).
    def _norm_contacts(t: torch.Tensor) -> set[tuple[int, int, int]]:
        out: set[tuple[int, int, int]] = set()
        for row in t[0].cpu().tolist():
            i, j, ttype = int(row[0]), int(row[1]), int(row[2])
            a, b = (i, j) if i < j else (j, i)
            out.add((a, b, ttype))
        return out
    a_set = _norm_contacts(orig.structure.token_contacts)
    b_set = _norm_contacts(rebuilt.structure.token_contacts)
    add(
        "structure.token_contacts (set)",
        a_set == b_set,
        detail="" if a_set == b_set else f"only_in_orig={a_set - b_set}, only_in_rebuilt={b_set - a_set}",
    )

    for k in ("element", "charge", "space_uid"):
        a = getattr(orig.reference, k).cpu()
        b = getattr(rebuilt.reference, k).cpu()
        ok = _eq_tensor(a, b)
        add(
            f"reference.{k}",
            ok,
            expected_diff=atom_count_diverges,
            detail="" if ok else f"shape orig={tuple(a.shape)} vs rebuilt={tuple(b.shape)}",
        )

    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """Roundtrip dataloader Batch <-> inference spec."""


def _common_item_options(func):
    options = [
        click.option("--index", type=int, help="Dataset index to fetch directly."),
        click.option("--name", type=str, help="Sample name like 7PTQ_1_1_."),
        click.option("--pdb-id", type=str),
        click.option("--assembly-id", type=str),
        click.option("--model-id", type=str),
        click.option("--alt-id", type=str),
        click.option("--chain-id", "chain_ids", multiple=True),
        click.option("--match", type=int, default=0),
        click.option("--seed", type=int, default=42),
        click.option("--epoch", type=int, default=0),
        click.option("--crop-indices", type=str, default=None),
        click.option(
            "--allow-fallback/--no-allow-fallback",
            default=False,
        ),
    ]
    decorated = func
    for o in reversed(options):
        decorated = o(decorated)
    return decorated


@cli.command()
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write fasta/a3m/spec.yaml into.",
)
@_common_item_options
@dataset_options
def dump(**kwargs: object) -> None:
    """Reverse a dataloader Batch into fasta + a3m + spec.yaml."""
    out_dir = cast("Path", kwargs.pop("out_dir"))
    args = _disable_cropping(ItemOptions(**cast("dict[str, Any]", kwargs)))
    ensure_paths_exist(
        [
            args.cif_db_path,
            args.a3m_db_path,
            args.edge_path,
            args.template_db_path,
            args.ccd_db_path,
        ],
    )
    dataset = build_dataset(args)
    batch, _, _, source, _ = fetch_direct_item(dataset, args)
    click.echo(f"Source: {source}")
    spec, chains = dump_inference_spec(
        batch,
        out_dir=out_dir,
        ccd_db_path=args.ccd_db_path,
        cif_db_path=args.cif_db_path,
        a3m_db_path=args.a3m_db_path,
        template_db_path=args.template_db_path,
    )
    click.echo(f"Wrote spec to {out_dir / 'spec.yaml'}")
    for c in chains:
        click.echo(
            f"  chain {c.chain_idx} Chain:{c.chain_letter} "
            f"entity={c.entity_fasta} n_res={c.token_hi - c.token_lo}",
        )


@cli.command()
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write fasta/a3m/spec.yaml into.",
)
@_common_item_options
@dataset_options
def roundtrip(**kwargs: object) -> None:
    """Dump + rebuild via build_inference_batch + diff against original Batch."""
    out_dir = cast("Path", kwargs.pop("out_dir"))
    args = _disable_cropping(ItemOptions(**cast("dict[str, Any]", kwargs)))
    max_msa_depth = args.max_msa_depth
    ensure_paths_exist(
        [
            args.cif_db_path,
            args.a3m_db_path,
            args.edge_path,
            args.template_db_path,
            args.ccd_db_path,
        ],
    )
    dataset = build_dataset(args)
    batch, _, _, source, _ = fetch_direct_item(dataset, args)
    click.echo(f"Source: {source}")
    spec, chains = dump_inference_spec(
        batch,
        out_dir=out_dir,
        ccd_db_path=args.ccd_db_path,
        cif_db_path=args.cif_db_path,
        a3m_db_path=args.a3m_db_path,
        template_db_path=args.template_db_path,
    )

    rebuilt = build_inference_batch(
        spec,
        max_msa_depth=max_msa_depth,
        missing_policy="query",
        seed=args.seed,
    )

    click.echo("=== Roundtrip diff ===")
    diffs = compare_batches(batch, rebuilt)
    real_fail = 0
    expected = 0
    for d in diffs:
        if d.ok:
            tag = "OK  "
        elif d.expected_diff:
            tag = "DIFF"  # informational divergence
            expected += 1
        else:
            tag = "FAIL"
            real_fail += 1
        click.echo(f"  [{tag}] {d.name}{(' — ' + d.detail) if d.detail else ''}")
    n_ok = len(diffs) - real_fail - expected
    click.echo(
        f"-> {n_ok}/{len(diffs)} strict matches, "
        f"{expected} expected divergences (canonical CCD vs cifmol), "
        f"{real_fail} unexpected failures",
    )


@cli.command(name="dense-contacts")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write fasta/a3m/spec.yaml into.",
)
@click.option(
    "--positive-cutoff",
    type=float,
    default=6.0,
    help="Distance threshold (Å) below which a token pair becomes a positive contact.",
)
@click.option(
    "--negative-cutoff",
    type=float,
    default=12.0,
    help="Distance threshold (Å) above which a token pair becomes a negative contact.",
)
@click.option(
    "--max-positive",
    type=int,
    default=None,
    help="Cap on positive pairs (random sub-sample). Default: keep all.",
)
@click.option(
    "--max-negative",
    type=int,
    default=None,
    help="Cap on negative pairs (random sub-sample). Default: keep all.",
)
@_common_item_options
@dataset_options
def dense_contacts(**kwargs: object) -> None:
    """Like ``dump`` but replaces contacts with all GT pairs under/over the cutoffs.

    Useful for stress-testing the contact module with a much higher
    contact density than ``to_token_contacts`` (Poisson-sampled ~20 per side).
    """
    out_dir = cast("Path", kwargs.pop("out_dir"))
    pos_cut = cast("float", kwargs.pop("positive_cutoff"))
    neg_cut = cast("float", kwargs.pop("negative_cutoff"))
    max_pos = cast("int | None", kwargs.pop("max_positive"))
    max_neg = cast("int | None", kwargs.pop("max_negative"))
    args = _disable_cropping(ItemOptions(**cast("dict[str, Any]", kwargs)))
    ensure_paths_exist(
        [
            args.cif_db_path,
            args.a3m_db_path,
            args.edge_path,
            args.template_db_path,
            args.ccd_db_path,
        ],
    )
    dataset = build_dataset(args)
    batch, _, _, source, _ = fetch_direct_item(dataset, args)
    click.echo(f"Source: {source}")

    chains = extract_chain_slices(batch)
    contacts = compute_dense_contacts(
        batch,
        chains,
        positive_cutoff=pos_cut,
        negative_cutoff=neg_cut,
        max_positive=max_pos,
        max_negative=max_neg,
        seed=args.seed,
    )
    click.echo(
        f"Dense contacts: positive={len(contacts.positive)} "
        f"negative={len(contacts.negative)} "
        f"(positive_cutoff={pos_cut} Å, negative_cutoff={neg_cut} Å)",
    )

    sample_name = str(batch.name[0])
    spec, chains = dump_inference_spec(
        batch,
        out_dir=out_dir,
        ccd_db_path=args.ccd_db_path,
        cif_db_path=args.cif_db_path,
        a3m_db_path=args.a3m_db_path,
        template_db_path=args.template_db_path,
        contacts_override=contacts,
        spec_name_override=f"dense_{sample_name}",
    )
    click.echo(f"Wrote spec to {out_dir / 'spec.yaml'}")
    for c in chains:
        click.echo(
            f"  chain {c.chain_idx} Chain:{c.chain_letter} "
            f"entity={c.entity_fasta} n_res={c.token_hi - c.token_lo}",
        )


def _read_multi_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a fasta file with one or more records: returns ``(header, body)`` per record."""
    records: list[tuple[str, str]] = []
    cur_header: str | None = None
    cur_body: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_header is not None:
                records.append((cur_header, "".join(cur_body)))
            cur_header = line[1:]  # strip the leading '>'
            cur_body = []
        else:
            cur_body.append(line)
    if cur_header is not None:
        records.append((cur_header, "".join(cur_body)))
    return records


@cli.command(name="from-fasta")
@click.option(
    "--query",
    "query_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Multi-record fasta file. Each '>' record becomes one chain in order.",
)
@click.option(
    "--a3m-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing per-chain a3m files (matched via --a3m-pattern).",
)
@click.option(
    "--a3m-pattern",
    type=str,
    default="*_{letter}.a3m",
    help="Glob pattern for per-chain a3m. ``{letter}`` is substituted with "
         "the auto-assigned chain letter (a, b, c, ...).",
)
@click.option(
    "--ccd-db-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to preprocessed CCD LMDB.",
)
@click.option(
    "--template-db-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to template LMDB (StructCooker-built, keyed by seq_id).",
)
@click.option(
    "--template-seq-ids",
    type=str,
    default=None,
    help="Comma-separated seq_ids per chain (in fasta order). Empty entries "
         "leave that chain without templates. Only used when --template-db-path is set.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write fasta/a3m/spec.yaml into.",
)
@click.option("--name", type=str, default=None, help="Optional spec name (default: query stem).")
def from_fasta(
    query_path: Path,
    a3m_dir: Path | None,
    a3m_pattern: str,
    ccd_db_path: Path,
    template_db_path: Path | None,
    template_seq_ids: str | None,
    out_dir: Path,
    name: str | None,
) -> None:
    """Convert a free-form multi-record fasta (+ optional a3m dir) into our spec.yaml layout.

    Chain letters are auto-assigned in order: 0->'a', 1->'b', ..., 25->'z',
    26->'aa', etc. Entity types are auto-detected from the residue letters
    (parens => ligand/branched, U => RNA, T => DNA, else protein).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_multi_fasta(query_path)
    if not records:
        msg = f"{query_path} contains no fasta records."
        raise click.ClickException(msg)

    seq_id_overrides: list[str] = []
    if template_seq_ids is not None:
        seq_id_overrides = [s.strip() for s in template_seq_ids.split(",")]

    chain_letters_map: dict[str, str] = {}
    fasta_paths: dict[str, str] = {}
    a3m_paths: dict[str, str] = {}
    template_map: dict[str, str] = {}
    summary: list[tuple[int, str, str, int]] = []

    for ci, (header, body) in enumerate(records):
        letter = index_to_lowercase_letter(ci)
        entity = _auto_detect_entity_type(body)
        chain_letters_map[str(ci)] = letter

        # Per-chain fasta: write the original header verbatim so the user can
        # still see their description; the relaxed parser handles it. Each
        # auto-assigned letter is unique (one per record), so letter-keying
        # the fasta dict can't collide here.
        fasta_path = out_dir / f"chain_{ci}_{letter}.fasta"
        fasta_path.write_text(f">{header}\n{body}\n")
        fasta_paths[letter] = str(fasta_path)
        summary.append((ci, letter, entity.value, len(body)))

        # a3m matching by chain letter
        if a3m_dir is not None and entity in (
            EntityType.PROTEIN, EntityType.RNA, EntityType.DNA,
        ):
            pattern = a3m_pattern.format(letter=letter)
            matches = sorted(a3m_dir.glob(pattern))
            if matches:
                src = matches[0]
                a3m_path = out_dir / f"chain_{ci}_{letter}.a3m"
                a3m_path.write_text(src.read_text())
                a3m_paths[letter] = str(a3m_path)

        # Template seq_id (per-chain, optional)
        if (
            template_db_path is not None
            and ci < len(seq_id_overrides)
            and seq_id_overrides[ci]
            and entity in (EntityType.PROTEIN, EntityType.RNA, EntityType.DNA)
        ):
            template_map[str(ci)] = seq_id_overrides[ci]

    spec_dict: dict[str, Any] = {
        "name": name or query_path.stem,
        "chain_letters": chain_letters_map,
        "fasta": fasta_paths,
        "ccd_db": str(ccd_db_path),
        "a3m": a3m_paths,
        "contacts": {"positive": [], "negative": []},
    }
    if template_db_path is not None:
        spec_dict["template_db"] = str(template_db_path)
        spec_dict["template"] = template_map
        spec_dict["template_n"] = 4

    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    for ci, letter, ev, n in summary:
        click.echo(
            f"  chain {ci} Chain:{letter} entity={ev} n_res={n}"
            + (" + a3m" if letter in a3m_paths else " (no a3m)")
            + (f" + tmpl[{template_map[str(ci)]}]" if str(ci) in template_map else ""),
        )
    click.echo(f"Wrote spec to {out_dir / 'spec.yaml'}")


if __name__ == "__main__":
    # Examples:
    #   python tests/test_inference_roundtrip.py dump \
    #       --name 7PTQ_1_1_. --chain-id A_1 \
    #       --tokenizer-level residue \
    #       --out-dir tests/inference_data/7PTQ
    #
    #   python tests/test_inference_roundtrip.py dense-contacts \
    #       --name 7PTQ_1_1_. --chain-id A_1 \
    #       --tokenizer-level residue \
    #       --positive-cutoff 8.0 --negative-cutoff 14.0 \
    #       --max-negative 2000 \
    #       --out-dir tests/inference_data/7PTQ_dense
    #
    #   python tests/test_inference_roundtrip.py from-fasta \
    #       --query tests/casp17/H1311/query.fasta \
    #       --a3m-dir tests/casp17/H1311 \
    #       --a3m-pattern '*_chains_{letter}.a3m' \
    #       --ccd-db-path /public_data02/CCD/preprocessed_CCD.lmdb \
    #       --out-dir tests/inference_data/H1311
    cli()
