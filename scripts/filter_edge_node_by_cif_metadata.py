"""Filter edge-node metadata rows using CIF-level chain/residue limits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Sequence


EDGE_REQUIRED_COLUMNS = (
    "cluster1",
    "cluster2",
    "pdb_id",
    "assembly_id",
    "model_id",
    "alt_id",
    "chain_id1",
    "chain_id2",
)
METADATA_REQUIRED_COLUMNS = (
    "cif_id",
    "chain_num",
    "residue_num",
)


@dataclass(frozen=True)
class CifMetadata:
    cif_id: str
    chain_num: int
    residue_num: int


@dataclass
class FilterStats:
    metadata_rows: int = 0
    metadata_kept: int = 0
    input_rows: int = 0
    written_rows: int = 0
    dropped_by_limit: int = 0
    dropped_missing_metadata: int = 0


def require_columns(
    fieldnames: Sequence[str] | None,
    required_columns: tuple[str, ...],
    *,
    label: str,
) -> list[str]:
    if fieldnames is None:
        msg = f"{label} is empty or missing a TSV header."
        raise click.ClickException(msg)

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        msg = f"{label} is missing required columns: {', '.join(missing_columns)}"
        raise click.ClickException(msg)
    return list(fieldnames)


def parse_int(value: str | None, *, column: str, row_name: str) -> int:
    if value is None or value == "":
        msg = f"{row_name} has empty {column}."
        raise click.ClickException(msg)
    try:
        return int(value)
    except ValueError as error:
        msg = f"{row_name} has non-integer {column}: {value!r}"
        raise click.ClickException(msg) from error


def normalize_cif_id(value: str) -> str:
    return value.strip().lower()


def edge_row_to_cif_id(row: dict[str, str]) -> str:
    return normalize_cif_id(
        "_".join(
            [
                row["pdb_id"],
                row["assembly_id"],
                row["model_id"],
                row["alt_id"],
            ],
        ),
    )


def load_cif_metadata(metadata_path: Path) -> dict[str, CifMetadata]:
    metadata: dict[str, CifMetadata] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_columns(reader.fieldnames, METADATA_REQUIRED_COLUMNS, label=str(metadata_path))

        for row in reader:
            cif_id = normalize_cif_id(row["cif_id"])
            if not cif_id:
                msg = f"{metadata_path} contains an empty cif_id."
                raise click.ClickException(msg)
            metadata[cif_id] = CifMetadata(
                cif_id=cif_id,
                chain_num=parse_int(row.get("chain_num"), column="chain_num", row_name=cif_id),
                residue_num=parse_int(
                    row.get("residue_num"),
                    column="residue_num",
                    row_name=cif_id,
                ),
            )
    return metadata


def metadata_passes_limits(
    metadata: CifMetadata,
    *,
    max_residue_num: int | None,
    max_chain_num: int | None,
) -> bool:
    return (
        (max_residue_num is None or metadata.residue_num <= max_residue_num)
        and (max_chain_num is None or metadata.chain_num <= max_chain_num)
    )


def validate_limits(max_residue_num: int | None, max_chain_num: int | None) -> None:
    if max_residue_num is None and max_chain_num is None:
        msg = "Set at least one of --max-residue-num or --max-chain-num."
        raise click.ClickException(msg)
    if max_residue_num is not None and max_residue_num < 1:
        msg = "--max-residue-num must be >= 1."
        raise click.ClickException(msg)
    if max_chain_num is not None and max_chain_num < 1:
        msg = "--max-chain-num must be >= 1."
        raise click.ClickException(msg)


def filter_edge_node_tsv(
    *,
    edge_node_path: Path,
    metadata_path: Path,
    output_path: Path,
    max_residue_num: int | None,
    max_chain_num: int | None,
    keep_missing_metadata: bool,
) -> FilterStats:
    validate_limits(max_residue_num, max_chain_num)

    metadata_by_cif_id = load_cif_metadata(metadata_path)
    passing_cif_ids = {
        cif_id
        for cif_id, metadata in metadata_by_cif_id.items()
        if metadata_passes_limits(
            metadata,
            max_residue_num=max_residue_num,
            max_chain_num=max_chain_num,
        )
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = FilterStats(
        metadata_rows=len(metadata_by_cif_id),
        metadata_kept=len(passing_cif_ids),
    )

    with (
        edge_node_path.open(newline="", encoding="utf-8") as input_handle,
        output_path.open("w", newline="", encoding="utf-8") as output_handle,
    ):
        reader = csv.DictReader(input_handle, delimiter="\t")
        fieldnames = require_columns(
            reader.fieldnames,
            EDGE_REQUIRED_COLUMNS,
            label=str(edge_node_path),
        )
        writer = csv.DictWriter(output_handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            stats.input_rows += 1
            cif_id = edge_row_to_cif_id(row)
            if cif_id not in metadata_by_cif_id:
                if keep_missing_metadata:
                    writer.writerow(row)
                    stats.written_rows += 1
                else:
                    stats.dropped_missing_metadata += 1
                continue

            if cif_id not in passing_cif_ids:
                stats.dropped_by_limit += 1
                continue

            writer.writerow(row)
            stats.written_rows += 1

    return stats


@click.command(context_settings={"show_default": True})
@click.option(
    "--edge-node-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Input train_edge_node TSV path.",
)
@click.option(
    "--metadata-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Input cif_metadata TSV path.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Output filtered edge-node TSV path.",
)
@click.option(
    "--max-residue-num",
    type=int,
    default=None,
    show_default="no limit",
    help="Keep rows whose CIF metadata residue_num is at most this value.",
)
@click.option(
    "--max-chain-num",
    type=int,
    default=None,
    show_default="no limit",
    help="Keep rows whose CIF metadata chain_num is at most this value.",
)
@click.option(
    "--keep-missing-metadata/--drop-missing-metadata",
    default=False,
    help="Keep edge rows whose CIF id is absent from cif_metadata.tsv.",
)
def main(
    *,
    edge_node_path: Path,
    metadata_path: Path,
    output_path: Path,
    max_residue_num: int | None,
    max_chain_num: int | None,
    keep_missing_metadata: bool,
) -> None:
    stats = filter_edge_node_tsv(
        edge_node_path=edge_node_path,
        metadata_path=metadata_path,
        output_path=output_path,
        max_residue_num=max_residue_num,
        max_chain_num=max_chain_num,
        keep_missing_metadata=keep_missing_metadata,
    )

    click.echo("=== Filter edge-node TSV ===")
    click.echo(f"Metadata rows          : {stats.metadata_rows}")
    click.echo(f"Metadata passing limit : {stats.metadata_kept}")
    click.echo(f"Input edge rows        : {stats.input_rows}")
    click.echo(f"Written edge rows      : {stats.written_rows}")
    click.echo(f"Dropped by limit       : {stats.dropped_by_limit}")
    click.echo(f"Dropped missing meta   : {stats.dropped_missing_metadata}")
    click.echo(f"Output                 : {output_path}")


if __name__ == "__main__":
    main()
