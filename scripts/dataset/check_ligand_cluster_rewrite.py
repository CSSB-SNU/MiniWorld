#!/usr/bin/env python3
"""Prove the ligand-cluster rewrite is a pure relabeling of an edge_node TSV.

Compares the identity-id TSV against the cluster-id one, row by row in order,
and asserts that nothing moved except the ligand cluster fields:

  1. same number of rows, same header;
  2. for every row, columns 3-8 (pdb_id, assembly_id, model_id, alt_id,
     chain_id1, chain_id2) are byte-identical;
  3. a non-``cL`` cluster field (``cP``/``cQ``/``cA``/``cD``/``cR``/``cN``/
     ``cB``/``cX``, or the ``None`` of a monomer row) is byte-identical;
  4. a ``cL`` cluster field equals exactly ``map[old]`` -- not merely "some other
     cL id", and not left behind when the map had an entry for it;
  5. no ligand field maps to a cluster that is not itself a representative, i.e.
     the map is idempotent -- ``map[map[x]] == map[x]``.

Checking the rewrite separately, rather than teaching the structural verifier
about the mapping, keeps that verifier at full strength: it compares per-ligand
cluster ids on both sides and so still catches a chain mix-up *within* a
chemical cluster, which a cluster-level comparison would silently accept.

Usage
-----
    python scripts/dataset/check_ligand_cluster_rewrite.py \\
        --identity .../train_edge_node_onlyPrtLig_ligand_identity.tsv \\
        --clustered .../train_edge_node_onlyPrtLig.tsv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

BASEDIR = Path("/public_data/bsoohyuncd/BioMolDB_20260224")
MAP_TSV = BASEDIR / "metadata/ligand_cluster_map_tanimoto50.tsv"
MAX_REPORTED = 10
LIGAND_PREFIX = "cL"
FIXED_COLUMNS = slice(2, 8)


def load_map(path: Path) -> dict[str, str]:
    """old ligand cluster id -> Tanimoto cluster id."""
    mapping: dict[str, str] = {}
    with path.open() as fh:
        next(fh)
        for line in fh:
            old, _, new = line.rstrip("\n").partition("\t")
            if old and new:
                mapping[old] = new
    return mapping


def main() -> None:  # noqa: C901, PLR0912
    """Compare the two TSVs and report any deviation from a pure relabeling."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--clustered", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=MAP_TSV)
    args = ap.parse_args()

    mapping = load_map(args.map)
    print(f"identity  : {args.identity}")
    print(f"clustered : {args.clustered}")
    print(f"map       : {args.map}  ({len(mapping):,} entries)")

    problems: list[str] = []
    counts: Counter[str] = Counter()

    def fail(msg: str) -> None:
        counts["problems"] += 1
        if len(problems) < MAX_REPORTED:
            problems.append(msg)

    # ---- 5: the map must be idempotent ----------------------------------
    non_idempotent = [k for k, v in mapping.items() if mapping.get(v, v) != v]
    if non_idempotent:
        fail(f"map is not idempotent for {len(non_idempotent)} ids, "
             f"e.g. {non_idempotent[:3]}")

    with args.identity.open() as fa, args.clustered.open() as fb:
        head_a, head_b = next(fa), next(fb)
        if head_a != head_b:
            fail(f"header differs: {head_a!r} vs {head_b!r}")
        for lineno, (la, lb) in enumerate(zip(fa, fb), start=2):
            counts["rows"] += 1
            a = la.rstrip("\n").split("\t")
            b = lb.rstrip("\n").split("\t")
            if len(a) != len(b):
                fail(f"line {lineno}: column count {len(a)} vs {len(b)}")
                continue
            if a[FIXED_COLUMNS] != b[FIXED_COLUMNS]:
                fail(f"line {lineno}: non-cluster columns changed "
                     f"{a[FIXED_COLUMNS]} -> {b[FIXED_COLUMNS]}")
            for i in (0, 1):
                old, new = a[i], b[i]
                if not old.startswith(LIGAND_PREFIX):
                    if old != new:
                        fail(f"line {lineno} col{i + 1}: non-ligand id changed "
                             f"{old} -> {new}")
                    else:
                        counts["nonligand_unchanged"] += 1
                    continue
                counts["ligand_fields"] += 1
                want = mapping.get(old, old)
                if new != want:
                    fail(f"line {lineno} col{i + 1}: {old} -> {new}, expected {want}")
                elif new != old:
                    counts["ligand_relabeled"] += 1
                else:
                    counts["ligand_already_rep"] += 1
        # length check
        extra_a = sum(1 for _ in fa)
        extra_b = sum(1 for _ in fb)
        if extra_a or extra_b:
            fail(f"row count differs: identity has {extra_a} extra, "
                 f"clustered has {extra_b} extra")

    print(f"\nrows compared           : {counts['rows']:,}")
    print(f"non-ligand fields intact: {counts['nonligand_unchanged']:,}")
    print(f"ligand fields           : {counts['ligand_fields']:,}")
    print(f"  relabeled correctly   : {counts['ligand_relabeled']:,}")
    print(f"  already representative: {counts['ligand_already_rep']:,}")

    if counts["problems"]:
        print(f"\nFAILED: {counts['problems']:,} problems, first {len(problems)}:")
        for p in problems:
            print("  ", p)
        raise SystemExit(f"{counts['problems']} problems")
    print("\nOK: pure relabeling -- only cL cluster fields changed, "
          "each to exactly map[old]")


if __name__ == "__main__":
    main()
