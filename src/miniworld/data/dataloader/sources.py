"""Turn external data sources into DataRecords and compute sampling weights.

Two kinds of source:
  * v2 manifest: items.tsv + resources.tsv (see _record_from_manifest_row).
  * v1 PDB compat: BioMolDBConfig.edge_id_to_bias_path (edge-id classified TSV).

After all sources are loaded, source_balanced_weights renormalizes the raw
item weights so that each source's total probability matches
``source_weights[source_name]`` regardless of item count.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from miniworld.configs.data import BioMolDBConfig, SamplerConfig
from miniworld.data.io import extract_lmdb_keys

from .types import BioMolDBV2Config, DataRecord, ResourceIndex


class _PDBEdgeRow:
    """One row of the legacy edge_id_to_bias TSV — used only inside this module."""

    def __init__(
        self,
        pdb_id: str,
        assembly_id: str,
        model_id: str,
        alt_id: str,
        chain_id1: str,
        chain_id2: str | None = None,
    ) -> None:
        self.pdb_id = pdb_id
        self.assembly_id = assembly_id
        self.model_id = model_id
        self.alt_id = alt_id
        self.chain_id1 = chain_id1
        self.chain_id2 = chain_id2


# ---------------------------------------------------------------------------
# v2 manifest TSV/CSV parsing
# ---------------------------------------------------------------------------


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


def _default_item_id(
    record_id: str,
    assembly_id: str,
    model_id: str,
    alt_id: str,
) -> str:
    return f"{record_id}:{assembly_id}:{model_id}:{alt_id}"


def resource_index(resources_path: Path) -> ResourceIndex:
    """Load feature_key/record_id to LMDB path mappings from resources.tsv."""
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

    msa_paths_by_chain = tuple(resources.msa.get(key, ()) for key in feature_keys)
    template_paths_by_chain = tuple(resources.template.get(key) for key in feature_keys)

    return DataRecord(
        item_id=_field(
            row,
            "item_id",
            default=_default_item_id(record_id, assembly_id, model_id, alt_id),
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
        msa_db_paths=msa_paths_by_chain,
        template_db_paths=template_paths_by_chain,
        weight=float(_field(row, "weight", default="1.0")),
        item_kind=_field(row, "item_kind", "type", default="unknown"),
        weight_group=_field(row, "weight_group", default="default"),
    )


def load_manifest_items(
    items_path: Path,
    resources_path: Path,
    validation: Literal["off", "paths", "keys"] = "off",
) -> tuple[list[DataRecord], list[float]]:
    """Read items.tsv into DataRecords using resources.tsv for LMDB paths.

    When ``validation != "off"``, checks the resources.tsv paths / shard keys
    at load time and drops items whose CIF record_id is missing from the CIF
    shard. See BioMolDBV2Config.manifest_validation for the mode contract.
    """
    resources = resource_index(resources_path)
    if validation != "off":
        _validate_manifest_paths(resources)

    records: list[DataRecord] = []
    weights: list[float] = []
    for row in _read_table(items_path):
        record = _record_from_manifest_row(row, resources)
        records.append(record)
        weights.append(record.weight)

    if validation == "keys":
        records, weights = _validate_manifest_keys(records, weights, resources)

    return records, weights


def _collect_resource_paths(resources: ResourceIndex) -> set[Path]:
    """All LMDB paths referenced by the manifest, across modalities."""
    paths: set[Path] = set()
    paths.update(resources.cif.values())
    for tup in resources.msa.values():
        paths.update(tup)
    paths.update(p for p in resources.template.values() if p is not None)
    return paths


def _validate_manifest_paths(resources: ResourceIndex) -> None:
    """Raise if any LMDB path referenced by the manifest does not exist."""
    missing = sorted(str(p) for p in _collect_resource_paths(resources) if not p.exists())
    if missing:
        preview = "\n  ".join(missing[:10])
        more = f"\n  ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        msg = (
            f"resources.tsv references {len(missing)} missing LMDB path(s):\n  "
            f"{preview}{more}"
        )
        raise FileNotFoundError(msg)


def _validate_manifest_keys(
    records: Sequence[DataRecord],
    weights: Sequence[float],
    resources: ResourceIndex,
) -> tuple[list[DataRecord], list[float]]:
    """Scan shards, drop items whose CIF key is missing, warn on MSA/template misses."""
    shard_keys: dict[Path, set[str]] = {}

    def keys_of(path: Path) -> set[str]:
        cache = shard_keys.get(path)
        if cache is None:
            cache = set(extract_lmdb_keys(path))
            shard_keys[path] = cache
        return cache

    kept_records: list[DataRecord] = []
    kept_weights: list[float] = []
    dropped: list[tuple[str, Path]] = []
    msa_misses = 0
    template_misses = 0

    for record, weight in zip(records, weights, strict=True):
        if record.record_id not in keys_of(record.cif_db_path):
            dropped.append((record.item_id, record.cif_db_path))
            continue

        for chain_idx, feature_key in enumerate(record.feature_keys):
            for msa_path in record.msa_db_paths[chain_idx] if chain_idx < len(record.msa_db_paths) else ():
                if feature_key not in keys_of(msa_path):
                    msa_misses += 1
            template_path_val = (
                record.template_db_paths[chain_idx]
                if chain_idx < len(record.template_db_paths)
                else None
            )
            if template_path_val is not None and feature_key not in keys_of(template_path_val):
                template_misses += 1

        kept_records.append(record)
        kept_weights.append(weight)

    print(  # noqa: T201  # deliberate init-time summary
        f"[manifest] validated {len(records)} items across {len(shard_keys)} shards: "
        f"dropped={len(dropped)}, msa_key_misses={msa_misses}, "
        f"template_key_misses={template_misses}",
    )
    if dropped:
        preview = "\n  ".join(f"{item_id}  (cif={path})" for item_id, path in dropped[:5])
        more = f"\n  ... (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
        print(f"[manifest] dropped items (record_id absent from cif shard):\n  {preview}{more}")  # noqa: T201

    return kept_records, kept_weights


# ---------------------------------------------------------------------------
# v1 legacy PDB edge_id_to_bias TSV parsing
# ---------------------------------------------------------------------------


_PDB_INTERFACE_TYPES = (
    "protein_protein",
    "protein_ligand",
    "protein_dna",
    "protein_rna",
    "antibody_protein",
    "dna_dna",
    "rna_rna",
    "dna_rna",
    "antibody_antibody",
    "antibody_ligand",
    "na_ligand",
    "etc_interface",
    "sole",
)


def _pdb_edge_type(edge_id: str) -> str:  # noqa: C901, PLR0911
    """Classify a legacy edge_id into the fixed sampler-weight bucket."""
    if "_" not in edge_id:
        return "sole"
    parse = set(re.findall(r"c([A-Z])", edge_id))

    if parse == {"P"}:
        return "protein_protein"
    if parse <= {"P", "L", "B"} and "P" in parse:
        return "protein_ligand"
    if parse == {"P", "D"}:
        return "protein_dna"
    if parse == {"P", "R"}:
        return "protein_rna"

    if parse == {"P", "A"}:
        return "antibody_protein"

    if parse == {"D"}:
        return "dna_dna"
    if parse == {"R"}:
        return "rna_rna"
    if parse == {"D", "R"}:
        return "dna_rna"

    if parse == {"A"}:
        return "antibody_antibody"
    if parse <= {"A", "L", "B"} and "A" in parse:
        return "antibody_ligand"
    if parse <= {"N", "L", "B"} and "N" in parse:
        return "na_ligand"
    return "etc_interface"


def read_pdb_edge_items(
    pdb_config: BioMolDBConfig,
    sampler_config: SamplerConfig,
) -> tuple[list[_PDBEdgeRow], list[float]]:
    """Read edge_id_to_bias TSV → (_PDBEdgeRow list, per-item sampler weight list)."""
    type_counts = dict.fromkeys(_PDB_INTERFACE_TYPES, 0)
    edge_id_to_items: dict[str, list[_PDBEdgeRow]] = {}

    with pdb_config.edge_id_to_bias_path.open("r") as f:
        for ii, _line in enumerate(f):
            if ii == 0:
                continue  # skip header
            line = _line.strip()
            if line == "":
                continue
            (
                cluster1,
                cluster2,
                pdb_id,
                assembly_id,
                model_id,
                alt_id,
                chain_id1,
                chain_id2,
            ) = line.split("\t")
            pdb_id = pdb_id.lower()  # to match cif_db keys
            if cluster2 == "None":
                edge_id = cluster1
                value = _PDBEdgeRow(pdb_id, assembly_id, model_id, alt_id, chain_id1)
            else:
                edge_id = f"{cluster1}_{cluster2}"
                value = _PDBEdgeRow(
                    pdb_id,
                    assembly_id,
                    model_id,
                    alt_id,
                    chain_id1,
                    chain_id2,
                )
            type_name = _pdb_edge_type(edge_id)
            if edge_id not in edge_id_to_items:
                edge_id_to_items[edge_id] = []
                # count unique clusters (edge_ids), not rows: the per-item
                # weight already divides by len(items), so normalizing by the
                # row count here would deflate each type's mass by its average
                # rows-per-cluster and break P(type) ∝ sampler weight.
                type_counts[type_name] += 1
            edge_id_to_items[edge_id].append(value)

    items: list[_PDBEdgeRow] = []
    weights: list[float] = []
    for edge_id, entries in edge_id_to_items.items():
        type_name = _pdb_edge_type(edge_id)
        w = getattr(sampler_config, type_name) / type_counts[type_name] / len(entries)
        weights.extend([w] * len(entries))
        items.extend(entries)
    return items, weights


# ---------------------------------------------------------------------------
# Unified train_item.tsv (edge_node style, source-tagged) parsing
# ---------------------------------------------------------------------------


class SourceDBs:
    """Per-source LMDB resources the loader routes a train_item row to."""

    def __init__(
        self,
        cif_db_path: Path,
        msa_db_paths: tuple[Path, ...],
        template_db_path: Path | None,
    ) -> None:
        self.cif_db_path = cif_db_path
        self.msa_db_paths = msa_db_paths
        self.template_db_path = template_db_path


def _edge_id(cluster1: str, cluster2: str) -> str:
    """Edge id used for PDB interface-type classification / cluster grouping."""
    return cluster1 if cluster2 == "None" else f"{cluster1}_{cluster2}"


def _train_item_weights(
    source: str,
    edge_id_to_rows: dict[str, list[int]],
    type_counts: dict[str, int],
    sampler_config: SamplerConfig,
    weights_out: list[float],
) -> None:
    """Fill ``weights_out`` for one source's rows with an AF3-style raw weight.

    * pdb: 3-tier — P(type) ∝ sampler weight, uniform over that type's clusters,
      uniform over a cluster's rows: ``W_type / type_clustercount / rowcount``.
    * distillation (monomer): cluster-uniform x instance-uniform — ``1 / rowcount``
      (the number of clusters normalizes out in ``source_balanced_weights``).
    """
    is_pdb = source == "pdb"
    for edge_id, row_indices in edge_id_to_rows.items():
        rowcount = len(row_indices)
        if is_pdb:
            type_name = _pdb_edge_type(edge_id)
            w = getattr(sampler_config, type_name) / type_counts[type_name] / rowcount
        else:
            w = 1.0 / rowcount
        for row_index in row_indices:
            weights_out[row_index] = w


def read_train_items(
    train_item_path: Path,
    source_dbs: dict[str, SourceDBs],
    sampler_config: SamplerConfig,
) -> tuple[list[DataRecord], list[float]]:
    """Read the unified train_item.tsv into DataRecords + AF3-style raw weights.

    Row schema (edge_node style, source-tagged):
        source cluster1 cluster2 pdb_id assembly_id model_id alt_id
        chain_id1 chain_id2
    Interface rows carry both clusters; monomer rows have ``cluster2 == "None"``.
    Keys are resolved at load time from the CIF (msa←seq_id, template←record+chain),
    so a record only carries the record id, chains, and per-source LMDB paths.
    """
    records: list[DataRecord] = []
    # Per-source grouping so weights are computed within each source.
    edge_rows: dict[str, dict[str, list[int]]] = {}
    type_counts: dict[str, dict[str, int]] = {}
    skipped_sources: dict[str, int] = {}

    with train_item_path.open("r") as handle:
        header = handle.readline()
        if not header.startswith("source"):
            msg = f"train_item.tsv must start with a 'source' header: {header!r}"
            raise ValueError(msg)
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped == "":
                continue
            (
                source,
                cluster1,
                cluster2,
                pdb_id,
                assembly_id,
                model_id,
                alt_id,
                chain_id1,
                chain_id2,
            ) = stripped.split("\t")
            dbs = source_dbs.get(source)
            if dbs is None:
                # Source not declared in this config (e.g. rna/disordered left
                # out): skip its rows rather than fail, so one train_item.tsv
                # serves configs that enable different source subsets.
                skipped_sources[source] = skipped_sources.get(source, 0) + 1
                continue

            record_id = pdb_id.lower() if source == "pdb" else pdb_id
            chain_ids = (chain_id1,) if cluster2 == "None" else (chain_id1, chain_id2)
            n_chains = len(chain_ids)

            row_index = len(records)
            records.append(
                DataRecord(
                    item_id=f"{source}:{record_id}:{assembly_id}:{model_id}:"
                    f"{alt_id}:{':'.join(chain_ids)}",
                    source=source,
                    record_id=record_id,
                    cif_db_path=dbs.cif_db_path,
                    assembly_id=assembly_id,
                    model_id=model_id,
                    alt_id=alt_id,
                    chain_ids=chain_ids,
                    feature_keys=(),  # msa/template keys resolved from the CIF
                    seq_ids=(),
                    msa_db_paths=(dbs.msa_db_paths,) * n_chains,
                    template_db_paths=(dbs.template_db_path,) * n_chains,
                    weight=1.0,
                    item_kind="interface" if cluster2 != "None" else "monomer",
                    weight_group=source,
                ),
            )

            edge_id = _edge_id(cluster1, cluster2)
            source_edges = edge_rows.setdefault(source, {})
            if edge_id not in source_edges:
                source_edges[edge_id] = []
                if source == "pdb":
                    counts = type_counts.setdefault(
                        source, dict.fromkeys(_PDB_INTERFACE_TYPES, 0),
                    )
                    counts[_pdb_edge_type(edge_id)] += 1
            source_edges[edge_id].append(row_index)

    if skipped_sources:
        summary = ", ".join(f"{s}={n}" for s, n in sorted(skipped_sources.items()))
        print(f"[train_item] skipped rows for unconfigured sources: {summary}")  # noqa: T201

    weights = [1.0] * len(records)
    for source, source_edges in edge_rows.items():
        _train_item_weights(
            source,
            source_edges,
            type_counts.get(source, {}),
            sampler_config,
            weights,
        )
    return records, weights


# ---------------------------------------------------------------------------
# Source-balanced weight computation
# ---------------------------------------------------------------------------


def configured_source_weights(config: BioMolDBV2Config) -> dict[str, float]:
    """Return source/db sampling weights with compatibility defaults."""
    weights = dict(config.source_weights)
    if config.pdb is not None:
        weights.setdefault("pdb", config.default_source_weight)
    for source in config.distillation_sources:
        weights.setdefault(source.name, source.weight)
    return weights


def _indices_by_source(records: Sequence[DataRecord]) -> dict[str, list[int]]:
    source_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        source_to_indices.setdefault(record.source, []).append(idx)
    return source_to_indices


def _active_source_weights(
    sources: Sequence[str],
    source_weights: dict[str, float],
    default_source_weight: float,
) -> dict[str, float]:
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
    weights = [float(weight) for weight in raw_weights]
    if any(weight < 0 for weight in weights):
        msg = f"Item weights for source '{source}' must be non-negative."
        raise ValueError(msg)

    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / weight_sum for weight in weights]


def source_balanced_weights(
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
