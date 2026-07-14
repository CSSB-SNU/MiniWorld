from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from biomol.core.types import BioMolDict
from biomol.core.utils import load_bytes
from pydantic import BaseModel
from torch.utils.data import DataLoader

from miniworld.configs.data import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.features import Batch, make_batch
from miniworld.data.io import extract_lmdb_keys, load_cifmol, load_raw_data
from miniworld.data.io.load import get_query_sequence
from miniworld.data.mols import CIFMolAttached, TemplateMol
from miniworld.data.pipeline import (
    MSA,
    ComplexMSA,
    ProteinTemplate,
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import remove_terminal_oxygen

from .dataloader import BioMolData, DataBias, WrongCroppingError, _bucketed_collate
from .sampler import WeightedSampler

if TYPE_CHECKING:
    from collections.abc import Sequence

class DistillationSourceConfig(BaseModel):
    """Configuration for one legacy-style distillation LMDB source."""

    name: str
    cif_db_path: Path
    a3m_db_paths: list[Path] = []
    template_db_path: Path | None = None
    weight: float = 1.0
    max_items: int | None = None
    assembly_id: str = "1"
    model_id: str = "1"
    alt_id: str = "."
    chain_ids: list[str] = ["1"]
    lookup_mode: Literal["record_id", "seq_id"] = "record_id"


class BioMolDBV2Config(BaseModel):
    """Database configuration for manifest-driven mixed training."""

    items_path: Path | None = None
    resources_path: Path | None = None
    ccd_preprocessed_path: Path | None = None

    # Source/db sampling probabilities. PDB still uses SamplerConfig inside this mass.
    source_weights: dict[str, float] = {}
    default_source_weight: float = 1.0
    sample_with_replacement: bool = True

    # Compatibility path while manifest generation is being introduced.
    pdb: BioMolDBConfig | None = None
    distillation_sources: list[DistillationSourceConfig] = []


@dataclass(frozen=True)
class DataRecord:
    """Resolved sampling item used by BioMolDataV2."""

    item_id: str
    source: str
    record_id: str
    cif_db_path: Path
    assembly_id: str
    model_id: str
    alt_id: str
    chain_ids: tuple[str, ...]
    feature_keys: tuple[str, ...]
    seq_ids: tuple[str, ...]
    msa_db_paths: tuple[tuple[Path, ...], ...]
    template_db_paths: tuple[Path | None, ...]
    weight: float = 1.0
    item_kind: str = "unknown"
    weight_group: str = "default"


@dataclass(frozen=True)
class ResourceIndex:
    """Lookup index from feature/record keys to resource LMDB paths."""

    cif: dict[str, Path]
    msa: dict[str, tuple[Path, ...]]
    template: dict[str, Path]


def _split_field(value: str | None) -> tuple[str, ...]:
    """Split comma/semicolon/pipe separated manifest fields."""
    if value is None:
        return ()
    value = value.strip()
    if value in {"", "None", "null"}:
        return ()
    for sep in (",", ";", "|"):
        if sep in value:
            return tuple(part.strip() for part in value.split(sep) if part.strip())
    return (value,)


def _read_table(path: Path) -> list[dict[str, str]]:
    """Read a TSV/CSV manifest file into row dictionaries."""
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def _field(row: dict[str, str], *names: str, default: str = "") -> str:
    """Read the first present manifest field name."""
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _resource_index(resources_path: Path) -> ResourceIndex:
    """Load feature_key/record_id to LMDB path mappings."""
    cif: dict[str, Path] = {}
    msa: dict[str, list[Path]] = {}
    template: dict[str, Path] = {}

    for row in _read_table(resources_path):
        present = _field(row, "present", default="1")
        if present in {"0", "False", "false", "no"}:
            continue
        key = _field(row, "feature_key", "record_id", "key")
        modality = _field(row, "modality", "kind", "resource_type").lower()
        path_text = _field(row, "db_path", "path")
        if key == "" or modality == "" or path_text == "":
            continue
        db_path = Path(path_text)
        if modality == "cif":
            cif[key] = db_path
        elif modality in {"msa", "a3m"}:
            msa.setdefault(key, []).append(db_path)
        elif modality == "template":
            template[key] = db_path

    return ResourceIndex(
        cif=cif,
        msa={key: tuple(paths) for key, paths in msa.items()},
        template=template,
    )


def _broadcast(values: tuple[str, ...], n: int, fallback: str) -> tuple[str, ...]:
    """Broadcast one manifest value to n chains."""
    if len(values) == n:
        return values
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) == 0:
        return (fallback,) * n
    msg = f"Cannot align {len(values)} values to {n} chains: {values}"
    raise ValueError(msg)


def _item_id(record_id: str, assembly_id: str, model_id: str, alt_id: str) -> str:
    return f"{record_id}:{assembly_id}:{model_id}:{alt_id}"


def _record_from_manifest_row(
    row: dict[str, str],
    resources: ResourceIndex,
) -> DataRecord:
    """Resolve one items.tsv row into a DataRecord."""
    record_id = _field(row, "record_id", "cif_key", "pdb_id")
    if record_id == "":
        msg = f"Manifest item row is missing record_id/pdb_id: {row}"
        raise ValueError(msg)

    assembly_id = _field(row, "assembly_id", default="1")
    model_id = _field(row, "model_id", default="1")
    alt_id = _field(row, "alt_id", default=".")
    chain_ids = _split_field(_field(row, "chain_ids", "crop_chain_ids", default="1"))
    if len(chain_ids) == 0:
        chain_ids = ("1",)

    feature_keys = _broadcast(
        _split_field(_field(row, "feature_keys", "feature_key")),
        len(chain_ids),
        record_id,
    )
    seq_ids = _broadcast(
        _split_field(_field(row, "seq_ids", "seq_id")),
        len(chain_ids),
        "",
    )

    cif_path_text = _field(row, "cif_db_path", "cif_path")
    cif_db_path = Path(cif_path_text) if cif_path_text else resources.cif.get(record_id)
    if cif_db_path is None:
        msg = f"No CIF resource for record_id '{record_id}'."
        raise KeyError(msg)

    msa_paths = tuple(resources.msa.get(key, ()) for key in feature_keys)
    template_paths = tuple(resources.template.get(key) for key in feature_keys)

    return DataRecord(
        item_id=_field(
            row,
            "item_id",
            default=_item_id(record_id, assembly_id, model_id, alt_id),
        ),
        source=_field(row, "source", default="manifest"),
        record_id=record_id,
        cif_db_path=cif_db_path,
        assembly_id=assembly_id,
        model_id=model_id,
        alt_id=alt_id,
        chain_ids=chain_ids,
        feature_keys=feature_keys,
        seq_ids=seq_ids,
        msa_db_paths=msa_paths,
        template_db_paths=template_paths,
        weight=float(_field(row, "weight", default="1.0")),
        item_kind=_field(row, "item_kind", "type", default="unknown"),
        weight_group=_field(row, "weight_group", default="default"),
    )


def _feature_index(record: DataRecord, chain_id: str) -> int:
    """Return the feature-key slot corresponding to a cropped chain."""
    try:
        return record.chain_ids.index(chain_id)
    except ValueError:
        if len(record.feature_keys) == 1:
            return 0
        msg = f"Chain '{chain_id}' is not present in item {record.item_id}."
        raise KeyError(msg) from None


def _feature_key(record: DataRecord, chain_id: str) -> str:
    return record.feature_keys[_feature_index(record, chain_id)]


def _msa_paths(record: DataRecord, chain_id: str) -> tuple[Path, ...]:
    return record.msa_db_paths[_feature_index(record, chain_id)]


def _template_path(record: DataRecord, chain_id: str) -> Path | None:
    return record.template_db_paths[_feature_index(record, chain_id)]


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


def _load_msa_v2(
    *,
    cifmol: CIFMolAttached,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    record: DataRecord,
    missing_policy: Literal["gap", "query"],
    pairing_mode: Literal["mixed", "paired_only", "no_pairing"],
) -> ComplexMSA:
    """Load MSA using chain-specific feature keys from the manifest."""
    msa_list: list[MSA] = []
    for chain_id, crop_indices in chain_id_to_crop_indices.items():
        if len(crop_indices) == 0:
            continue
        key = _feature_key(record, chain_id)
        msa = _load_a3m_from_paths(key, _msa_paths(record, chain_id))
        if msa is None:
            query_seq = get_query_sequence(cifmol, chain_id)
            msa = MSA.from_query(query_sequence=query_seq, seq_id=key)
        else:
            msa = MSA.cropped(msa, crop_indices)
        msa_list.append(msa)

    return ComplexMSA(
        MSAs=msa_list,
        missing_policy=missing_policy,
        pairing_mode=pairing_mode,
    )


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

    return ProteinTemplate(template_list=templates, ids=[template_id] * len(templates))


def _load_templates_v2(
    *,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    record: DataRecord,
    n_templates: int,
    rng: np.random.Generator | None,
) -> ProteinTemplate:
    """Load template features using chain-specific feature keys."""
    templates_list: list[ProteinTemplate] = []
    template_id = 0
    for chain_id, crop_indices in chain_id_to_crop_indices.items():
        template_env = _template_path(record, chain_id)
        if template_env is None or len(crop_indices) == 0:
            templates_list.append(
                ProteinTemplate(n_residues=crop_indices.shape[0], ids=[template_id]),
            )
            template_id += 1
            continue

        templates = _load_template_by_key(
            key=_feature_key(record, chain_id),
            template_id=template_id,
            env_path=template_env,
            crop_indices=crop_indices,
            n_templates=n_templates,
            rng=rng,
        )
        templates_list.append(templates)
        template_id += 1
    return ProteinTemplate.concat(templates_list)


def _configured_source_weights(config: BioMolDBV2Config) -> dict[str, float]:
    """Return source/db sampling weights with compatibility defaults."""
    weights = dict(config.source_weights)
    if config.pdb is not None:
        weights.setdefault("pdb", config.default_source_weight)
    for source in config.distillation_sources:
        weights.setdefault(source.name, source.weight)
    return weights


def _indices_by_source(records: Sequence[DataRecord]) -> dict[str, list[int]]:
    """Group record indices by source/db name."""
    source_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        source_to_indices.setdefault(record.source, []).append(idx)
    return source_to_indices


def _active_source_weights(
    sources: Sequence[str],
    source_weights: dict[str, float],
    default_source_weight: float,
) -> dict[str, float]:
    """Resolve positive source weights for the loaded sources only."""
    active: dict[str, float] = {}
    for source in sources:
        source_weight = source_weights.get(source, default_source_weight)
        if source_weight < 0:
            msg = f"Source weight for '{source}' must be non-negative."
            raise ValueError(msg)
        if source_weight > 0:
            active[source] = source_weight
    return active


def _normalized_raw_weights(source: str, raw_weights: Sequence[float]) -> list[float]:
    """Normalize item weights inside one source/db."""
    weights = [float(weight) for weight in raw_weights]
    if any(weight < 0 for weight in weights):
        msg = f"Item weights for source '{source}' must be non-negative."
        raise ValueError(msg)

    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / weight_sum for weight in weights]


def _source_balanced_weights(
    records: Sequence[DataRecord],
    raw_weights: Sequence[float],
    source_weights: dict[str, float],
    default_source_weight: float,
) -> list[float]:
    """Normalize item weights so sampling is source-first, then item/type inside."""
    if len(records) != len(raw_weights):
        msg = f"Got {len(records)} records but {len(raw_weights)} weights."
        raise ValueError(msg)
    if len(records) == 0:
        return []

    source_to_indices = _indices_by_source(records)
    active_weights = _active_source_weights(
        tuple(source_to_indices),
        source_weights,
        default_source_weight,
    )
    active_weight_sum = sum(active_weights.values())
    if active_weight_sum <= 0:
        msg = "At least one loaded source must have a positive sampling weight."
        raise ValueError(msg)

    weights = [0.0] * len(records)
    for source, indices in source_to_indices.items():
        source_weight = active_weights.get(source)
        if source_weight is None:
            continue

        group_weights = _normalized_raw_weights(
            source,
            [raw_weights[idx] for idx in indices],
        )
        source_probability = source_weight / active_weight_sum
        for idx, group_weight in zip(indices, group_weights, strict=True):
            weights[idx] = source_probability * group_weight

    return weights


class BioMolDataV2(BioMolData):
    """Dataset that consumes unified item/resource manifests."""

    class BioMolConfig(BaseModel):  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuration for BioMolDataV2."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        template_config: TemplateConfig = TemplateConfig()
        DB_config: BioMolDBV2Config
        tokenizer_config: TokenizerConfig = TokenizerConfig()
        sampler_config: SamplerConfig = SamplerConfig()

    config: BioMolConfig
    items: list[DataRecord]  # pyright: ignore[reportIncompatibleVariableOverride]

    def _load_items(self) -> None:
        self.items = []
        self.weights = []
        if self.config.DB_config.items_path and self.config.DB_config.resources_path:
            self._load_manifest_items(
                self.config.DB_config.items_path,
                self.config.DB_config.resources_path,
            )
        else:
            self._load_compat_items()

        self.weights = _source_balanced_weights(
            records=self.items,
            raw_weights=self.weights,
            source_weights=_configured_source_weights(self.config.DB_config),
            default_source_weight=self.config.DB_config.default_source_weight,
        )

    def _load_manifest_items(self, items_path: Path, resources_path: Path) -> None:
        resources = _resource_index(resources_path)
        for row in _read_table(items_path):
            record = _record_from_manifest_row(row, resources)
            self.items.append(record)
            self.weights.append(record.weight)

    def _load_compat_items(self) -> None:
        if self.config.DB_config.pdb is not None:
            self._load_pdb_items(self.config.DB_config.pdb)
        self._load_distillation_items()

    def _load_pdb_items(self, pdb_config: BioMolDBConfig) -> None:
        edge_items, edge_weights = self._read_legacy_pdb_items(pdb_config)
        for bias, weight in zip(edge_items, edge_weights, strict=True):
            chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])
            self.items.append(
                DataRecord(
                    item_id=f"pdb:{bias.pdb_id}:{bias.assembly_id}:"
                    f"{bias.model_id}:{bias.alt_id}:{':'.join(chain_ids)}",
                    source="pdb",
                    record_id=bias.pdb_id,
                    cif_db_path=pdb_config.cif_db_path,
                    assembly_id=bias.assembly_id,
                    model_id=bias.model_id,
                    alt_id=bias.alt_id,
                    chain_ids=tuple(chain_ids),
                    feature_keys=(),
                    seq_ids=(),
                    msa_db_paths=(),
                    template_db_paths=(),
                    weight=weight,
                    weight_group="pdb",
                ),
            )
            self.weights.append(weight)

    def _read_legacy_pdb_items(
        self,
        pdb_config: BioMolDBConfig,
    ) -> tuple[list[DataBias], list[float]]:
        original_db_config = self.config.DB_config
        original_items = self.items
        original_weights = self.weights
        try:
            self.config.DB_config = pdb_config  # type: ignore[assignment]
            super()._load_items()
            return list(self.items), list(self.weights)  # type: ignore[list-item]
        finally:
            self.config.DB_config = original_db_config  # type: ignore[assignment]
            self.items = original_items  # pyright: ignore[reportIncompatibleVariableOverride]
            self.weights = original_weights

    def _load_distillation_items(self) -> None:
        for source in self.config.DB_config.distillation_sources:
            keys = extract_lmdb_keys(source.cif_db_path, max_keys=source.max_items)
            if len(keys) == 0:
                continue

            per_item_weight = 1.0
            for key in keys:
                n_chains = len(source.chain_ids)
                self.items.append(
                    DataRecord(
                        item_id=f"{source.name}:{key}",
                        source=source.name,
                        record_id=key,
                        cif_db_path=source.cif_db_path,
                        assembly_id=source.assembly_id,
                        model_id=source.model_id,
                        alt_id=source.alt_id,
                        chain_ids=tuple(source.chain_ids),
                        feature_keys=(key,) * n_chains
                        if source.lookup_mode == "record_id"
                        else (),
                        seq_ids=(),
                        msa_db_paths=(tuple(source.a3m_db_paths),) * n_chains,
                        template_db_paths=(source.template_db_path,) * n_chains,
                        weight=per_item_weight,
                        weight_group=source.name,
                    ),
                )
                self.weights.append(per_item_weight)

    def _load_ccd_preprocessed(self) -> None:
        ccd_path = self.config.DB_config.ccd_preprocessed_path
        if ccd_path is None and self.config.DB_config.pdb is not None:
            ccd_path = self.config.DB_config.pdb.ccd_preprocessed_path
        if ccd_path is None:
            msg = "CCD preprocessed path is not provided in the v2 config."
            raise ValueError(msg)

        keys = extract_lmdb_keys(ccd_path)
        from .dataloader import FragmentedCCDMolCache

        self.fragmented_ccd_mols = FragmentedCCDMolCache(ccd_path, keys)

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        rng = np.random.default_rng()
        record = self.items[idx]

        while True:
            try:
                return self.get_item_by_record(record=record, rng=rng)
            except WrongCroppingError:  # noqa: PERF203
                idx = int(rng.integers(0, len(self)))
                record = self.items[idx]

    def _record_with_runtime_feature_keys(
        self,
        record: DataRecord,
        cifmol: CIFMolAttached,
        chain_ids: Sequence[str] | None = None,
    ) -> DataRecord:
        """Fill feature keys for compatibility records from CIF chain seq_ids."""
        resolved_chain_ids = tuple(chain_ids) if chain_ids is not None else record.chain_ids
        if (
            record.chain_ids == resolved_chain_ids
            and len(record.feature_keys) == len(resolved_chain_ids)
        ):
            return record

        if record.source == "pdb":
            keys = []
            for chain_id in resolved_chain_ids:
                seq_id = cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value
                keys.append(str(seq_id))
        elif len(record.feature_keys) == 1:
            keys = list(record.feature_keys) * len(resolved_chain_ids)
        else:
            keys = []
            for chain_id in resolved_chain_ids:
                keys.append(_feature_key(record, chain_id))

        if record.source == "pdb" and self.config.DB_config.pdb is not None:
            pdb = self.config.DB_config.pdb
            msa_paths = ((pdb.a3m_db_path,),) * len(keys)
            template_paths = (pdb.template_db_path,) * len(keys)
        elif len(record.msa_db_paths) == 1 and len(resolved_chain_ids) > 1:
            msa_paths = record.msa_db_paths * len(resolved_chain_ids)
            template_paths = record.template_db_paths * len(resolved_chain_ids)
        else:
            msa_paths = record.msa_db_paths
            template_paths = record.template_db_paths

        return DataRecord(
            item_id=record.item_id,
            source=record.source,
            record_id=record.record_id,
            cif_db_path=record.cif_db_path,
            assembly_id=record.assembly_id,
            model_id=record.model_id,
            alt_id=record.alt_id,
            chain_ids=resolved_chain_ids,
            feature_keys=tuple(keys),
            seq_ids=tuple(keys),
            msa_db_paths=msa_paths,
            template_db_paths=template_paths,
            weight=record.weight,
            item_kind=record.item_kind,
            weight_group=record.weight_group,
        )

    def get_item_by_record(
        self,
        *,
        record: DataRecord,
        crop_indices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> Batch:
        """Build a batch from an already resolved record."""
        if rng is None:
            rng = np.random.default_rng()
        cifmol = load_cifmol(
            db_path=record.cif_db_path,
            pdb_id=record.record_id,
            assembly_id=record.assembly_id,
            model_id=record.model_id,
            alt_id=record.alt_id,
        )
        record = self._record_with_runtime_feature_keys(record, cifmol)

        if crop_indices is None:
            (
                crop_indices,
                chain_id_to_crop_indices,
                atom_to_token_idx_map,
                token_to_residue_idx_map,
                _focus,
            ) = self.get_crop_indices(
                cifmol=cifmol,
                chain_ids=list(record.chain_ids),
                max_tokens=self.config.crop_config.max_tokens,
                max_atoms=self.config.crop_config.max_atoms,
                rng=rng,
            )
            if crop_indices.shape[0] == 0:
                msg = (
                    f"Failed to crop {record.record_id}_"
                    f"{record.assembly_id}_{record.model_id}_{record.alt_id}."
                )
                raise WrongCroppingError(msg)
            record = self._record_with_runtime_feature_keys(
                record,
                cifmol,
                tuple(chain_id_to_crop_indices),
            )
        else:
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
            valid_xyz = cifmol.atoms.xyz.value
            valid_mask = np.isfinite(valid_xyz).all(axis=1)
            focus = (
                valid_xyz[valid_mask][rng.integers(0, valid_mask.sum())]
                if valid_mask.any()
                else np.zeros(3)
            )
            atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(
                cifmol,
                focus=focus,
                fragmented_ccd_mols=self.fragmented_ccd_mols,
                config=self.config.tokenizer_config.dynamic_config,
            )
            record = self._record_with_runtime_feature_keys(
                record,
                cifmol,
                tuple(chain_id_to_crop_indices),
            )

        cifmol = cifmol.residues[crop_indices].extract()
        atom_mask = remove_terminal_oxygen(cifmol)
        cifmol = cifmol.atoms[atom_mask].extract()
        atom_to_token_idx_map = atom_to_token_idx_map[atom_mask]

        complex_msa = _load_msa_v2(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            record=record,
            missing_policy=self.config.msa_config.missing_policy,
            pairing_mode=self.config.msa_config.pairing_mode,
        )
        msa = sample_msa(
            msa=complex_msa,
            max_msa_depth=self.config.msa_config.max_msa_depth,
            rng=rng,
        )

        templates = _load_templates_v2(
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            record=record,
            n_templates=self.config.template_config.n_templates,
            rng=rng,
        )

        return make_batch(
            cifmol=cifmol,
            msa=msa,
            templates=templates,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
            rng=rng,
        )

    def create_ddp_dataloader(
        self,
        rank: int,
        *,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        bucket_msa_multiple: int | None = None,
        bucket_token_multiple: int | None = None,
        bucket_atom_multiple: int | None = None,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with WeightedSampler."""
        self.seed = int(seed)

        sampler = WeightedSampler(
            dataset=self,
            weights=self.weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            replacement=self.config.DB_config.sample_with_replacement,
        )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})
        if num_workers == 0:
            kwargs.pop("prefetch_factor", None)

        params = {
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": False,
            "multiprocessing_context": ("spawn" if num_workers > 0 else None),
            "collate_fn": functools.partial(
                _bucketed_collate,
                bucket_msa_multiple=bucket_msa_multiple,
                bucket_token_multiple=bucket_token_multiple,
                bucket_atom_multiple=bucket_atom_multiple,
            ),
        }
        params.update(kwargs)
        return DataLoader(self, **params)
