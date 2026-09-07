#!/usr/bin/env python3
"""Rewrite an edge_node TSV, swapping ligand cluster ids for Tanimoto clusters.

Only the two cluster columns are touched, and only where they hold a ``cL`` id.
``cluster1`` and ``cluster2`` are both mapped, because a ligand can sit on either
side of an edge.  Every other column -- pdb_id, assembly_id, model_id, alt_id,
chain_id1, chain_id2 -- is copied through byte for byte, as is any non-ligand
cluster id (``cP``, ``cQ``, ``cA``, ``cD``, ``cR``, ``cN``, ``cB``, ``cX``) and
the literal ``None`` of a monomer row.  Row order is preserved.

Why
---
``seq_cluster30.tsv`` gives every ligand its own single-member cluster, so
``dataloader._load_items`` -- which groups rows by ``edge_id =
f"{cluster1}_{cluster2}"`` and hands each edge_id equal sampling mass within its
type -- treats a congeneric series of 130 analogues as 130 independent clusters.
``cluster_ligands_tanimoto.py`` builds the real clustering; this script writes it
into the edge table so the sampler sees it without any dataloader change.

Usage
-----
    python scripts/dataset/apply_ligand_clusters_to_edges.py \\
        --in-tsv  .../train_edge_node_onlyPrtLig.tsv \\
        --out-tsv .../train_edge_node_onlyPrtLig_ligclust.tsv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

BASEDIR = Path("/public_data/bsoohyuncd/BioMolDB_20260224")
MAP_TSV = BASEDIR / "metadata/ligand_cluster_map_tanimoto50.tsv"

TSV_HEADER = (
    "cluster1\tcluster2\tpdb_id\tassembly_id\tmodel_id\talt_id\tchain_id1\tchain_id2\n"
)
LIGAND_PREFIX = "cL"
N_COLUMNS = 8


def load_map(path: Path) -> dict[str, str]:
    """old ligand cluster id -> Tanimoto cluster id."""
    mapping: dict[str, str] = {}
    with path.open() as fh:
        header = next(fh).rstrip("\n").split("\t")
        if header != ["old_cluster", "new_cluster"]:
            msg = f"{path}: unexpected header {header}"
            raise SystemExit(msg)
        for line in fh:
            old, _, new = line.rstrip("\n").partition("\t")
            if old and new:
                mapping[old] = new
    return mapping


def main() -> None:
    """Rewrite the cluster columns and report what changed."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-tsv", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=MAP_TSV)
    args = ap.parse_args()

    mapping = load_map(args.map)
    print(f"in  : {args.in_tsv}")
    print(f"out : {args.out_tsv}")
    print(f"map : {args.map}  ({len(mapping):,} ligand clusters)")

    counts: Counter[str] = Counter()
    unmapped: set[str] = set()
    edge_before: set[str] = set()
    edge_after: set[str] = set()

    tmp = args.out_tsv.with_suffix(args.out_tsv.suffix + ".tmp")
    with args.in_tsv.open() as fin, tmp.open("w") as fout:
        header = next(fin)
        if header != TSV_HEADER:
            msg = f"{args.in_tsv}: unexpected header {header!r}"
            raise SystemExit(msg)
        fout.write(header)
        for line in fin:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != N_COLUMNS:
                msg = f"{args.in_tsv}: expected {N_COLUMNS} columns, got {len(fields)}"
                raise SystemExit(msg)
            counts["rows"] += 1
            edge_before.add(f"{fields[0]}_{fields[1]}")
            for i in (0, 1):
                cluster = fields[i]
                if not cluster.startswith(LIGAND_PREFIX):
                    continue
                counts["ligand_fields"] += 1
                new = mapping.get(cluster)
                if new is None:
                    unmapped.add(cluster)
                    continue
                if new != cluster:
                    counts["ligand_fields_changed"] += 1
                    fields[i] = new
            edge_after.add(f"{fields[0]}_{fields[1]}")
            fout.write("\t".join(fields) + "\n")
    tmp.replace(args.out_tsv)

    print(f"\nrows written              : {counts['rows']:,}")
    print(f"ligand cluster fields     : {counts['ligand_fields']:,}")
    print(f"  rewritten               : {counts['ligand_fields_changed']:,}")
    print(f"  already representative  : "
          f"{counts['ligand_fields'] - counts['ligand_fields_changed'] - 0:,}")
    print(f"distinct edge_id before   : {len(edge_before):,}")
    print(f"distinct edge_id after    : {len(edge_after):,}")
    if edge_before:
        print(f"  collapse                : "
              f"{100 * (1 - len(edge_after) / len(edge_before)):.1f}% fewer sampling units")
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} ligand clusters had no mapping, left unchanged:")
        print(f"  {sorted(unmapped)[:10]}")
    print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
