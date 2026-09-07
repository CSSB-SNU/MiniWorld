#!/usr/bin/env python3
"""Dump sub-entries of a cif_attached LMDB to mmCIF for eyeballing.

Written to check the ``*_onlyPrtDna`` subsets: it loads entries through the same
path the dataloader uses (``load_bytes`` -> ``CIFMolAttached.from_dict``), so a
file appearing here proves the stored blob still decodes, and the per-chain table
printed alongside it shows the composition the entry filter claims to guarantee.

Usage
-----
    # first 3 keys named by the edge_node TSV, all their sub-entries
    pixi run python scripts/dump_prt_dna_cif.py \
        --lmdb  /public_data/.../cif_attached_valid_2_onlyPrtDna.lmdb \
        --tsv   /public_data/.../metadata/valid2_edge_node_onlyPrtDna.tsv \
        --out   /tmp/prt_dna_cif --limit 3

    # named entries only
    pixi run python scripts/dump_prt_dna_cif.py --lmdb ... --pdb 1a1f 6ymw
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes
from structcooker.mols import CIFMolAttached
from structcooker.mols.utils import to_cif
from structcooker.utils.mapping import cluster_maps, polymer_entity_types

# Cluster-type letters that must not appear.  Default suits a protein-DNA set
# (RNA + hybrid); pass --forbid DN when dumping a protein-RNA set.
DEFAULT_FORBID = "RN"


def read_tsv_keys(tsv_path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Return pdb ids in TSV order plus the sub-entry keys each one references."""
    order: list[str] = []
    seen: set[str] = set()
    subkeys: dict[str, set[str]] = {}
    with tsv_path.open() as fh:
        next(fh)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                continue
            pdb = fields[2].lower()
            if pdb not in seen:
                seen.add(pdb)
                order.append(pdb)
                subkeys[pdb] = set()
            subkeys[pdb].add(f"{fields[3]}_{fields[4]}_{fields[5]}")
    return order, subkeys


def atom_group_labels(cifmol: CIFMolAttached) -> list[str]:
    """ATOM / HETATM per written atom line, decided by the chain's entity type.

    ``residues.hetero`` cannot be used: in mmCIF that flag means microheterogeneity
    and BioMolDB carries 0 for every residue, ligands included -- the same finding
    ``inf_foldbench/scripts/fix_hetatm_flags.py`` documents.  Non-polymer, branched
    and unknown chains therefore become HETATM by entity type instead.

    The order matches ``to_cif``, which drops atoms whose xyz is NaN.
    """
    xyz = np.asarray(cifmol.atoms.xyz.value)
    mask = ~np.isnan(xyz).any(axis=1)
    res_to_chain = np.asarray(cifmol.index_table.res_to_chain)
    atom_to_res = np.asarray(cifmol.index_table.atom_to_res)
    entity_type = np.asarray(cifmol.chains.entity_type.value)
    per_atom = entity_type[res_to_chain[atom_to_res]][mask]
    return ["ATOM" if str(t) in polymer_entity_types else "HETATM" for t in per_atom]


def normalise_cif(path: Path, block: str, groups: list[str]) -> tuple[int, int]:
    """Make ``structcooker.mols.utils.to_cif`` output loadable by a mmCIF reader.

    That writer emits two things a strict parser rejects, both unrelated to which
    entries a subset holds: the data block name is the repr of ``cifmol.id``
    (``data_['7Q94']_1_.``), and ``_atom_site.group_PDB`` carries the raw
    ``residues.hetero`` flag as ``0``/``1`` rather than ``ATOM``/``HETATM``.
    Only those two fields are touched; every coordinate line is left as produced.

    Returns
    -------
    (n_atom, n_hetatm)
    """
    lines = path.read_text().splitlines()
    n_atom = n_hetatm = 0
    row = 0
    for i, line in enumerate(lines):
        if line.startswith("data_"):
            lines[i] = f"data_{block}"
            continue
        head, _, rest = line.partition(" ")
        if head not in {"0", "1"}:
            continue
        group = groups[row]
        row += 1
        if group == "ATOM":
            n_atom += 1
        else:
            n_hetatm += 1
        lines[i] = f"{group:<6s} " + rest
    if row != len(groups):
        msg = f"{path}: rewrote {row} atom lines but the mol has {len(groups)}"
        raise ValueError(msg)
    path.write_text("\n".join(lines) + "\n")
    return n_atom, n_hetatm


def describe(cifmol: CIFMolAttached) -> tuple[str, Counter]:
    """One-line-per-chain table plus a cluster-type tally."""
    chain_id = np.asarray(cifmol.chains.chain_id.value)
    cluster_id = np.asarray(cifmol.chains.cluster_id.value)
    entity_type = np.asarray(cifmol.chains.entity_type.value)
    res_to_chain = np.asarray(cifmol.index_table.res_to_chain)
    atom_to_res = np.asarray(cifmol.index_table.atom_to_res)
    atom_to_chain = res_to_chain[atom_to_res]

    tally: Counter[str] = Counter()
    lines = [
        f"    {'chain':8s} {'cluster':10s} {'type':16s} {'entity_type':32s} "
        f"{'n_res':>6s} {'n_atom':>7s}",
    ]
    for idx, chain in enumerate(chain_id):
        letter = str(cluster_id[idx])[1]
        tally[letter] += 1
        n_res = int((res_to_chain == idx).sum())
        n_atom = int((atom_to_chain == idx).sum())
        lines.append(
            f"    {chain!s:8s} {cluster_id[idx]!s:10s} "
            f"{cluster_maps.get(letter, '?'):16s} {entity_type[idx]!s:32s} "
            f"{n_res:6d} {n_atom:7d}",
        )
    return "\n".join(lines), tally


def main() -> None:  # noqa: C901, PLR0912
    """Dump the requested entries and report their chain composition."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb", required=True, type=Path, help="cif_attached LMDB to read.")
    ap.add_argument("--tsv", type=Path, help="edge_node TSV naming which keys to dump.")
    ap.add_argument("--pdb", nargs="*", default=[], help="Explicit pdb ids (lowercase).")
    ap.add_argument(
        "--pick",
        nargs="*",
        default=[],
        help="Exact sub-entries as pdb:assembly_model_alt, e.g. 8ua7:1_1_. -- one file each.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Directory for the .cif files.")
    ap.add_argument("--limit", type=int, default=3, help="How many pdb entries to dump.")
    ap.add_argument(
        "--all-subentries",
        action="store_true",
        help="Dump every sub-entry, not just those the TSV references.",
    )
    ap.add_argument(
        "--forbid",
        default=DEFAULT_FORBID,
        help=(
            "Cluster-type letters that must not appear, e.g. RN for a protein-DNA "
            f"set or DN for a protein-RNA set (default: {DEFAULT_FORBID})."
        ),
    )
    ap.add_argument(
        "--raw-cif",
        action="store_true",
        help="Skip the data-block / group_PDB normalisation; write to_cif output as-is.",
    )
    args = ap.parse_args()
    forbidden = frozenset(args.forbid)

    if args.pick:
        subkeys = {}
        targets = []
        for spec in args.pick:
            pdb, sep, asm_key = spec.partition(":")
            if not sep:
                raise SystemExit(f"--pick wants pdb:asm_key, got {spec!r}")
            if pdb not in subkeys:
                subkeys[pdb] = set()
                targets.append(pdb)
            subkeys[pdb].add(asm_key)
    elif args.pdb:
        targets, subkeys = list(args.pdb), {}
    elif args.tsv:
        order, subkeys = read_tsv_keys(args.tsv)
        targets = order[: args.limit]
    else:
        raise SystemExit("pass --tsv, --pdb or --pick")

    args.out.mkdir(parents=True, exist_ok=True)
    grand: Counter[str] = Counter()
    n_files = 0
    violations: list[str] = []

    env = lmdb.open(str(args.lmdb), readonly=True, lock=False)
    print(f"lmdb {args.lmdb} holds {env.stat()['entries']} entries")
    with env.begin() as txn:
        raws = {pdb: txn.get(pdb.encode()) for pdb in targets}
    env.close()

    for pdb in targets:
        raw = raws[pdb]
        if raw is None:
            print(f"\n{pdb}: NOT IN LMDB")
            continue
        entry = load_bytes(bytes(raw))
        wanted = sorted(entry) if (args.all_subentries or pdb not in subkeys) else sorted(
            subkeys[pdb],
        )
        print(f"\n{'=' * 78}\n{pdb}  sub-entries in blob: {sorted(entry)}")
        print(f"{'dumping':>9s}: {wanted}")

        for asm_key in wanted:
            item = entry.get(asm_key)
            if item is None:
                print(f"  {asm_key}: MISSING from blob")
                continue
            cifmol = CIFMolAttached.from_dict(item["cifmol_attached_dict"])
            table, tally = describe(cifmol)
            grand.update(tally)
            bad = set(tally) & forbidden
            flag = f"  <-- FORBIDDEN {sorted(bad)}" if bad else ""
            if bad:
                violations.append(f"{pdb} {asm_key} {sorted(bad)}")
            comp = " ".join(f"{cluster_maps.get(k, k)}x{v}" for k, v in sorted(tally.items()))
            print(f"\n  {asm_key}: {len(cifmol.chains)} chains | {comp}{flag}")
            print(table)

            path = args.out / f"{pdb}_{asm_key}.cif"
            to_cif(cifmol, path)
            n_files += 1
            note = ""
            if not args.raw_cif:
                n_a, n_h = normalise_cif(
                    path, f"{pdb.upper()}_{asm_key}", atom_group_labels(cifmol),
                )
                note = f", {n_a} ATOM + {n_h} HETATM"
            print(f"    -> {path} ({path.stat().st_size / 1024:.0f} KB{note})")

    print(f"\n{'=' * 78}")
    print(f"wrote {n_files} cif files to {args.out}")
    print("chain cluster types over everything dumped:")
    for letter, count in sorted(grand.items(), key=lambda kv: -kv[1]):
        print(f"  {cluster_maps.get(letter, letter):16s} ({letter}): {count}")
    if violations:
        print("\nFORBIDDEN chain types found:")
        for v in violations:
            print(" ", v)
    else:
        names = ", ".join(cluster_maps.get(c, c) for c in sorted(forbidden))
        print(f"\nno forbidden chain ({names}) in any dumped sub-entry")


if __name__ == "__main__":
    main()
