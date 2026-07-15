"""Build a resources.tsv manifest for BioMolData by scanning shard LMDBs.

The resulting manifest maps each feature_key to the single shard that holds it,
so BioMolData (in manifest mode) skips the runtime try-each-shard fallback.

Example:
    pixi run python scripts/build_resources_manifest.py \\
        --a3m /nhn/msa/shard_0.lmdb /nhn/msa/shard_1.lmdb /nhn/msa/shard_2.lmdb \\
        --template /nhn/template.lmdb \\
        --cif /nhn/cif.lmdb \\
        --output /nhn/resources.tsv

Then in BioMolDBV2Config: point ``resources_path`` at the emitted TSV and
``items_path`` at the corresponding items.tsv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from miniworld.data.io import extract_lmdb_keys


def scan_shards(
    modality: str,
    shard_paths: Sequence[Path],
    writer: csv.DictWriter,
    dedup: set[str] | None,
) -> tuple[int, int]:
    """Write one manifest row per (feature_key, shard). Returns (rows, unique_keys)."""
    n_rows = 0
    unique_keys: set[str] = set()
    for shard in shard_paths:
        resolved = str(shard.resolve())
        for key in extract_lmdb_keys(shard):
            if dedup is not None and key in dedup:
                continue
            if dedup is not None:
                dedup.add(key)
            writer.writerow(
                {
                    "feature_key": key,
                    "modality": modality,
                    "db_path": resolved,
                    "present": "1",
                },
            )
            n_rows += 1
            unique_keys.add(key)
    return n_rows, len(unique_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--a3m",
        nargs="*",
        type=Path,
        default=[],
        help="MSA/a3m shard LMDBs (emitted as modality=a3m).",
    )
    parser.add_argument("--template", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help=(
            "By default the first shard containing a key wins (per modality). "
            "Pass this to keep every hit and let the loader treat later shards "
            "as fallback candidates."
        ),
    )
    args = parser.parse_args()

    if not (args.cif or args.a3m or args.template):
        parser.error("At least one of --cif/--a3m/--template is required.")

    for shard in (*args.cif, *args.a3m, *args.template):
        if not shard.exists():
            parser.error(f"Shard does not exist: {shard}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature_key", "modality", "db_path", "present"],
            delimiter="\t",
        )
        writer.writeheader()

        counts: dict[str, tuple[int, int]] = {}
        for modality, shards in (
            ("cif", args.cif),
            ("a3m", args.a3m),
            ("template", args.template),
        ):
            if not shards:
                continue
            dedup: set[str] | None = None if args.allow_duplicates else set()
            counts[modality] = scan_shards(modality, shards, writer, dedup)

    total_rows = sum(rows for rows, _ in counts.values())
    print(f"Wrote {total_rows} rows → {args.output}")
    for modality, (rows, uniq) in counts.items():
        note = f"{rows} rows, {uniq} unique keys"
        if rows != uniq:
            note += f" ({rows - uniq} duplicates kept as fallback)"
        print(f"  {modality:9} {note}")


if __name__ == "__main__":
    main()
