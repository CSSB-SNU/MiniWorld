#!/usr/bin/env python3
"""Verify the protein-ligand pair built by ``build_prt_lig_set.py``.

The build makes three claims, and each is checked against the structures rather
than against the builder's own bookkeeping:

  1. every crystallisation-aid molecule is gone from the coordinates;
  2. no kept entry contains a nucleic-acid chain, a nucleotide, or a nucleotide
     analogue -- anywhere in any of its sub-entries, whether or not a row names
     it.  The vocabulary is rebuilt here by calling the builder's own
     ``nucleotide_codes``, so the test is over the same 2,650 codes;
  3. no surviving row pairs a protein with a ligand covalently bonded to it.

and three invariants that say nothing ELSE was touched:

  3. the output rows are a subset of the source rows, field for field -- so the
     ligand cluster ids are still the source's own ids, not a relabelling;
  4. every non-aid chain the source had is still in the output -- ligand chains
     that no row names must NOT have been stripped, which is where this build
     deliberately differs from the previous one;
  5. rows and entries agree: every row's entry, sub-entry and both chain ids
     exist, and every LMDB key is named by at least one row.

``AID_CODES`` is imported from the builder so the two lists cannot drift apart.
Covalency, by contrast, is recomputed here from scratch -- a direct per-row
distance and a direct struct_conn scan, not the builder's one-tree-per-sub-entry
shortcut -- so a bug in that optimisation would show up as a disagreement rather
than be reproduced.

Usage
-----
    PYTHONPATH=/home/bsoohyuncd/software/StructCooker/src \\
    /home/bsoohyuncd/software/StructCooker/.pixi/envs/default/bin/python \\
        scripts/dataset/verify_prt_lig_set.py --split train [--sample 3000|--full]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes
from structcooker.mols import CIFMolAttached

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_prt_lig_set import (  # noqa: E402
    AID_CODES,
    CCD_LMDB,
    COVALENT_ANGSTROM,
    LIGAND_TYPE,
    NUCLEIC_TYPES,
    PROTEIN_TYPES,
    SPLITS,
    TSV_HEADER,
    nucleotide_codes,
)

MAX_REPORT = 10


def read_tsv(path: Path) -> list[tuple[str, ...]]:
    """Every row of an edge_node TSV, header checked."""
    rows: list[tuple[str, ...]] = []
    with path.open() as fh:
        if next(fh).rstrip("\n") != TSV_HEADER.rstrip("\n"):
            msg = f"unexpected header in {path}"
            raise SystemExit(msg)
        for line in fh:
            fields = tuple(line.rstrip("\n").split("\t"))
            if len(fields) == 8:
                rows.append(fields)
    return rows


def covale_pairs(cif_dict: dict, atom_chain: np.ndarray) -> set[frozenset[int]]:
    """Chain-index pairs joined by a ``covale`` struct_conn edge."""
    conn = cif_dict.get("atoms", {}).get("edges", {}).get("struct_conn")
    if conn is None:
        return set()
    value = np.asarray(conn["value"])
    src = np.asarray(conn["src_indices"])
    dst = np.asarray(conn["dst_indices"])
    out: set[frozenset[int]] = set()
    for row in range(len(src)):
        if "covale" not in str(value[row]):
            continue
        a, b = int(src[row]), int(dst[row])
        if 0 <= a < len(atom_chain) and 0 <= b < len(atom_chain):
            out.add(frozenset((int(atom_chain[a]), int(atom_chain[b]))))
    return out


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run every check and report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=[*SPLITS], default="train")
    ap.add_argument("--in-lmdb", type=Path)
    ap.add_argument("--in-tsv", type=Path)
    ap.add_argument("--out-lmdb", type=Path)
    ap.add_argument("--out-tsv", type=Path)
    ap.add_argument("--sample", type=int, default=3000,
                    help="PDB entries to open for the structural checks.")
    ap.add_argument("--full", action="store_true",
                    help="Open every entry instead of a sample.")
    ap.add_argument("--covalent-angstrom", type=float, default=COVALENT_ANGSTROM)
    ap.add_argument("--ccd-lmdb", type=Path, default=CCD_LMDB)
    ap.add_argument("--nucleotide-tier", default="similar")
    ap.add_argument("--coverage", type=float, default=0.7)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-bits", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = dict(SPLITS[args.split])
    for key, value in (("in_lmdb", args.in_lmdb), ("in_tsv", args.in_tsv),
                       ("out_lmdb", args.out_lmdb), ("out_tsv", args.out_tsv)):
        if value is not None:
            paths[key] = value

    problems: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not ok:
            problems.append(label)

    print(f"source lmdb : {paths['in_lmdb']}")
    print(f"source tsv  : {paths['in_tsv']}")
    print(f"output lmdb : {paths['out_lmdb']}")
    print(f"output tsv  : {paths['out_tsv']}")
    print(f"aid codes   : {' '.join(sorted(AID_CODES))}")

    out_rows = read_tsv(paths["out_tsv"])
    print(f"\noutput rows : {len(out_rows):,}")

    # ---- 1. every row is a protein-ligand row ---------------------------
    print("\n== row shape")
    bad_type = [r for r in out_rows
                if LIGAND_TYPE not in {r[0][1], r[1][1]}
                or not {r[0][1], r[1][1]} & PROTEIN_TYPES]
    check(not bad_type, "every row is cL against cP/cQ/cA",
          f"{len(bad_type):,} bad" if bad_type else "")
    dupes = len(out_rows) - len(set(out_rows))
    check(dupes == 0, "no duplicate rows", f"{dupes:,} duplicates" if dupes else "")

    aid_set = set(AID_CODES.tolist())
    by_pdb: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for row in out_rows:
        by_pdb[row[2].lower()].append(row)
    print(f"  ....  {len(by_pdb):,} distinct pdb entries named by rows")

    # ---- 2. keys and rows agree -----------------------------------------
    print("\n== lmdb / tsv agreement")
    env_out = lmdb.open(str(paths["out_lmdb"]), readonly=True, lock=False,
                        max_readers=2048)
    with env_out.begin() as txn:
        keys = {bytes(k).decode() for k, _ in txn.cursor()}
    print(f"  ....  {len(keys):,} keys in the output lmdb")
    missing_key = sorted(set(by_pdb) - keys)
    orphan_key = sorted(keys - set(by_pdb))
    check(not missing_key, "every row's pdb entry is in the lmdb",
          f"{len(missing_key):,} missing e.g. {missing_key[:MAX_REPORT]}"
          if missing_key else "")
    check(not orphan_key, "every lmdb key is named by a row",
          f"{len(orphan_key):,} orphaned e.g. {orphan_key[:MAX_REPORT]}"
          if orphan_key else "")

    # ---- 3. rows are a subset of the source ------------------------------
    targets = sorted(by_pdb)
    if not args.full:
        rng = random.Random(args.seed)
        targets = sorted(rng.sample(targets, min(args.sample, len(targets))))
    print(f"\n== structural checks on {len(targets):,} entries"
          f"{' (full)' if args.full else ' (sampled)'}")

    want = set(targets)
    src_rows: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    with paths["in_tsv"].open() as fh:
        next(fh)
        for line in fh:
            fields = tuple(line.rstrip("\n").split("\t"))
            if len(fields) == 8 and fields[2].lower() in want:
                src_rows[fields[2].lower()].add(fields)

    not_subset = 0
    for pdb in targets:
        for row in by_pdb[pdb]:
            if row not in src_rows[pdb]:
                not_subset += 1
    check(not_subset == 0,
          "output rows appear verbatim in the source tsv (ids not relabelled)",
          f"{not_subset:,} rows not found" if not_subset else "")

    # ---- 4/5/6. open the structures --------------------------------------
    env_in = lmdb.open(str(paths["in_lmdb"]), readonly=True, lock=False,
                       max_readers=2048)
    print("\n  building the nucleotide vocabulary to re-test exclusion ...")
    nuc_codes, _ = nucleotide_codes(
        args.ccd_lmdb, args.nucleotide_tier, args.coverage, args.similarity,
        args.fp_radius, args.fp_bits)
    nuc_set = set(nuc_codes.tolist())

    stats: Counter[str] = Counter()
    aid_left: list[str] = []
    nucleic_left: list[str] = []
    covalent_left: list[str] = []
    chain_lost: list[str] = []
    row_absent: list[str] = []
    started = time.time()

    with env_out.begin(buffers=True) as tout, env_in.begin(buffers=True) as tin:
        for n_done, pdb in enumerate(targets, start=1):
            raw_out = tout.get(pdb.encode())
            if raw_out is None:
                continue
            entry_out = load_bytes(bytes(raw_out))
            raw_in = tin.get(pdb.encode())
            entry_in = load_bytes(bytes(raw_in)) if raw_in is not None else {}

            rows_by_key: dict[str, list] = defaultdict(list)
            for row in by_pdb[pdb]:
                rows_by_key[f"{row[3]}_{row[4]}_{row[5]}"].append(row)

            for asm_key, items in rows_by_key.items():
                if asm_key not in entry_out:
                    row_absent.append(f"{pdb}:{asm_key} sub-entry missing")
                    continue
                cif = entry_out[asm_key]["cifmol_attached_dict"]
                mol = CIFMolAttached.from_dict(cif)
                chain_id = np.asarray(mol.chains.chain_id.value).astype(str)
                cluster_id = np.asarray(mol.chains.cluster_id.value).astype(str)
                where = {name: i for i, name in enumerate(chain_id)}
                res_ccd = np.asarray(mol.residues.chem_comp_id.value).astype(str)
                stats["subentries"] += 1

                # -- aids gone
                still = sorted(set(res_ccd.tolist()) & aid_set)
                if still:
                    aid_left.append(f"{pdb}:{asm_key} {still}")

                # -- no nucleic acid survived anywhere in a kept entry
                na_chains = sorted({c for c in cluster_id.tolist()
                                    if c[1:2] in NUCLEIC_TYPES})
                if na_chains:
                    nucleic_left.append(f"{pdb}:{asm_key} chains {na_chains[:3]}")
                na_res = sorted(set(res_ccd.tolist()) & nuc_set)
                if na_res:
                    nucleic_left.append(f"{pdb}:{asm_key} residues {na_res[:5]}")

                # -- non-aid chains preserved
                if asm_key in entry_in:
                    mol_in = CIFMolAttached.from_dict(
                        entry_in[asm_key]["cifmol_attached_dict"])
                    cin = np.asarray(mol_in.chains.chain_id.value).astype(str)
                    rin = np.asarray(mol_in.residues.chem_comp_id.value).astype(str)
                    r2c = np.asarray(mol_in.index_table.res_to_chain)
                    n_res = np.bincount(r2c, minlength=len(cin))
                    n_aid = np.bincount(r2c, weights=np.isin(rin, AID_CODES).astype(float),
                                        minlength=len(cin))
                    survivors = {cin[i] for i in range(len(cin))
                                 if not (n_res[i] > 0 and n_aid[i] == n_res[i])}
                    lost = sorted(survivors - set(chain_id.tolist()))
                    if lost:
                        chain_lost.append(f"{pdb}:{asm_key} lost {lost[:5]}")
                    stats["chains_source"] += len(cin)
                    stats["chains_kept"] += len(chain_id)

                # -- rows resolve, and are not covalent
                atom_chain = np.asarray(mol.index_table.res_to_chain)[
                    np.asarray(mol.index_table.atom_to_res)]
                xyz = np.asarray(mol.atoms.xyz.value)
                pairs = covale_pairs(cif, atom_chain)
                for row in items:
                    if row[6] not in where or row[7] not in where:
                        row_absent.append(f"{pdb}:{asm_key} {row[6]}/{row[7]}")
                        continue
                    i6, i7 = where[row[6]], where[row[7]]
                    lig, prot = ((i6, i7) if cluster_id[i6][1] == LIGAND_TYPE
                                 else (i7, i6))
                    stats["rows_checked"] += 1
                    if frozenset((lig, prot)) in pairs:
                        covalent_left.append(f"{pdb}:{asm_key} {row[6]}/{row[7]} covale")
                        continue
                    a = xyz[atom_chain == lig]
                    b = xyz[atom_chain == prot]
                    a = a[np.isfinite(a).all(1)]
                    b = b[np.isfinite(b).all(1)]
                    if len(a) == 0 or len(b) == 0:
                        continue
                    dmin = float(np.sqrt(
                        ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())
                    if dmin < args.covalent_angstrom:
                        covalent_left.append(
                            f"{pdb}:{asm_key} {row[6]}/{row[7]} {dmin:.2f}A")

            if n_done % 500 == 0:
                rate = n_done / max(time.time() - started, 1e-9)
                print(f"  ... {n_done:,}/{len(targets):,} ({rate:.0f}/s)", flush=True)
    env_in.close()
    env_out.close()

    print(f"\n  ....  {stats['subentries']:,} sub-entries, "
          f"{stats['rows_checked']:,} rows re-tested "
          f"({time.time() - started:.0f}s)")

    check(not aid_left, "no crystallisation aid remains in the coordinates",
          f"{len(aid_left):,} e.g. {aid_left[:MAX_REPORT]}" if aid_left else "")
    check(not nucleic_left,
          "no kept entry contains a nucleic-acid chain or nucleotide/analogue",
          f"{len(nucleic_left):,} e.g. {nucleic_left[:MAX_REPORT]}"
          if nucleic_left else "")
    check(not covalent_left, "no surviving row is protein-covalent ligand",
          f"{len(covalent_left):,} e.g. {covalent_left[:MAX_REPORT]}"
          if covalent_left else "")
    check(not chain_lost, "every non-aid chain of the source survives",
          f"{len(chain_lost):,} e.g. {chain_lost[:MAX_REPORT]}" if chain_lost else "")
    check(not row_absent, "every row's sub-entry and both chains exist",
          f"{len(row_absent):,} e.g. {row_absent[:MAX_REPORT]}" if row_absent else "")
    if stats["chains_source"]:
        print(f"  ....  chains: {stats['chains_kept']:,} kept of "
              f"{stats['chains_source']:,} in the source "
              f"({100 * stats['chains_kept'] / stats['chains_source']:.1f}%)")

    print(f"\n{'=' * 68}")
    if problems:
        print(f"FAILED {len(problems)} check(s):")
        for p in problems:
            print("  -", p)
        raise SystemExit(f"{len(problems)} failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
