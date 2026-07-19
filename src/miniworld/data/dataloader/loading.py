"""LMDB-backed loaders: CCD fragmentation cache, MSA, template."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from biomol.core.types import BioMolDict
from biomol.core.utils import load_bytes

from miniworld.data.io import load_raw_data
from miniworld.data.io.load import get_query_sequence
from miniworld.data.mols import CCDMol, CIFMolAttached, FragmentedCCDMol, TemplateMol
from miniworld.data.pipeline import (
    MSA,
    ComplexMSA,
    ProteinTemplate,
    fragment_ccdmol_all_merges,
)

from .types import DataRecord, msa_paths, template_path


# ---------------------------------------------------------------------------
# CCD fragmentation cache (lazy LMDB-backed)
# ---------------------------------------------------------------------------


class FragmentedCCDMolCache(Mapping[str, dict[int, FragmentedCCDMol]]):
    """Lazy cache for CCD fragmentations keyed by chemcomp id."""

    def __init__(self, ccd_preprocessed_path: Path, keys: list[str]) -> None:
        self.ccd_preprocessed_path = ccd_preprocessed_path
        self._keys = set(keys)
        self._cache: dict[str, dict[int, FragmentedCCDMol]] = {}

    def __getitem__(self, key: str) -> dict[int, FragmentedCCDMol]:
        if key in self._cache:
            return self._cache[key]
        if key not in self._keys:
            raise KeyError(key)

        data = load_raw_data(key, self.ccd_preprocessed_path)
        if data is None:
            raise KeyError(key)

        ccdmol = CCDMol.from_bytes(data)
        fragments = fragment_ccdmol_all_merges(ccdmol)
        self._cache[key] = fragments
        return fragments

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._cache or key in self._keys)


# ---------------------------------------------------------------------------
# MSA loading
# ---------------------------------------------------------------------------


def _first_raw_data(key: str, env_paths: Sequence[Path]) -> bytes | None:
    """Return the first LMDB value found for key across env_paths."""
    for env_path in env_paths:
        raw = load_raw_data(key, env_path)
        if raw is not None:
            return raw
    return None


def _load_a3m_from_paths(key: str, env_paths: Sequence[Path]) -> MSA | None:
    """Load one MSA by key from one of several LMDB paths."""
    raw = _first_raw_data(key, env_paths)
    if raw is None:
        return None
    msa_dict = load_bytes(bytes(raw))["msa_dict"]
    return MSA(
        seq_id=key,
        sequences=msa_dict["sequences"],
        headers=msa_dict["headers"],
    )


def chain_seq_id(cifmol: CIFMolAttached, chain_id: str) -> str:
    """Return a chain's MSA lookup key: its per-sequence ``seq_id`` from the CIF.

    MSAs are keyed by seq_id (identical sequences deduped), so the key is read
    from the loaded, attached CIF rather than carried on the item.
    """
    return str(cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value)


def load_record_msa(
    *,
    cifmol: CIFMolAttached,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    record: DataRecord,
    missing_policy: Literal["gap", "query"],
    pairing_mode: Literal["mixed", "paired_only", "no_pairing"],
) -> ComplexMSA:
    """Load MSA by each chain's seq_id (read from the CIF) from the record's shards."""
    msa_list: list[MSA] = []
    for chain_id, crop_indices in chain_id_to_crop_indices.items():
        if len(crop_indices) == 0:
            continue
        seq_id = chain_seq_id(cifmol, chain_id)
        msa = _load_a3m_from_paths(seq_id, msa_paths(record, chain_id))
        if msa is None:
            query_seq = get_query_sequence(cifmol, chain_id)
            msa = MSA.from_query(query_sequence=query_seq, seq_id=seq_id)
        else:
            msa = MSA.cropped(msa, crop_indices)
        msa_list.append(msa)

    return ComplexMSA(
        MSAs=msa_list,
        missing_policy=missing_policy,
        pairing_mode=pairing_mode,
    )


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def _load_template_by_key(
    *,
    key: str,
    template_id: int,
    env_path: Path,
    crop_indices: np.ndarray,
    n_templates: int,
    rng: np.random.Generator | None,
    res_min: int = 4,
) -> ProteinTemplate:
    """Load templates by an explicit lookup key."""
    raw = load_raw_data(key, env_path)
    if raw is None:
        return ProteinTemplate(n_residues=len(crop_indices), ids=[template_id])

    template_mols = load_bytes(bytes(raw))["template_mols"]
    ids = list(template_mols.keys())
    if rng is not None:
        rng.shuffle(ids)

    templates: list[TemplateMol] = []
    for cif_key in ids:
        item = cast("BioMolDict", template_mols[cif_key])
        template = TemplateMol.from_dict(item)
        template = template.residues[crop_indices].extract()

        atom_xyz = np.asarray(template.atoms.xyz.value, dtype=float).reshape(
            len(template.residues),
            4,
            3,
        )
        valid_residue_mask = np.asarray(
            np.isfinite(atom_xyz[:, :3, :]).all(axis=(-1, -2)),
            dtype=bool,
        )
        if int(valid_residue_mask.sum()) < res_min:
            continue

        templates.append(template)
        if len(templates) >= n_templates:
            break

    if len(templates) == 0:
        return ProteinTemplate(n_residues=len(crop_indices), ids=[template_id])

    return ProteinTemplate(
        template_list=templates,
        ids=[template_id] * len(templates),
    )


def chain_template_key(cifmol: CIFMolAttached, record_id: str, chain_id: str) -> str:
    """Return a chain's template lookup key: ``{RECORD}_{label_chain}``.

    The template DB (and all upstream cif.fasta / cif_chain / seq_id_map data) is
    keyed by the **label** chain id, not auth_asym_id. The DB key drops the copy
    suffix, so cif label ``A_1`` → ``A`` (key ``101M_A``) and distillation label
    ``1`` → ``1`` (key ``MGYP..._1``). ``record_id`` is used verbatim: all keys
    (cif + template) are lower-cased, so PDB cif ``100d`` → template ``100d_A``
    and distillation ``MGYP...`` → ``MGYP..._1`` (already lower-case-consistent).

    Using auth_asym_id here was a bug: for multi-chain PDB entries label≠auth
    (e.g. label ``C_1`` has auth ``F``), so it fetched a different chain's
    template — wrong length → IndexError in ``template.residues[crop_indices]``.
    ``cifmol`` is kept in the signature for call-site compatibility but unused.
    """
    del cifmol  # label keying needs no CIF lookup
    label = chain_id.rsplit("_", 1)[0]
    return f"{record_id}_{label}"


def load_record_templates(
    *,
    cifmol: CIFMolAttached,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    record: DataRecord,
    n_templates: int,
    rng: np.random.Generator | None,
) -> ProteinTemplate:
    """Load template features by each chain's ``{RECORD}_{auth}`` key.

    When ``n_templates <= 0`` the LMDB read is skipped entirely and each chain
    gets an empty ``ProteinTemplate`` — models that don't consume templates
    (e.g. MiniSWAModel's pair-only trunk) then avoid paying the biomol
    ``load_bytes`` deserialize cost, which was ~55% of preprocess wall time
    at the default ``n_templates=4``.
    """
    if n_templates <= 0:
        return ProteinTemplate.concat(
            [
                ProteinTemplate(n_residues=crop_indices.shape[0], ids=[template_id])
                for template_id, crop_indices in enumerate(
                    chain_id_to_crop_indices.values(),
                )
            ],
        )

    templates_list: list[ProteinTemplate] = []
    template_id = 0
    for chain_id, crop_indices in chain_id_to_crop_indices.items():
        template_env = template_path(record, chain_id)
        if template_env is None or len(crop_indices) == 0:
            templates_list.append(
                ProteinTemplate(n_residues=crop_indices.shape[0], ids=[template_id]),
            )
            template_id += 1
            continue

        templates = _load_template_by_key(
            key=chain_template_key(cifmol, record.record_id, chain_id),
            template_id=template_id,
            env_path=template_env,
            crop_indices=crop_indices,
            n_templates=n_templates,
            rng=rng,
        )
        templates_list.append(templates)
        template_id += 1
    return ProteinTemplate.concat(templates_list)
