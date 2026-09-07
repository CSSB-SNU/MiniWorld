#!/usr/bin/env python3
"""Carve the protein-DNA-only subset out of the BioMolDB train / valid2 sets.

Target definition
-----------------
Three filters, applied in this order.

  1. *Aid removal* -- every residue whose ``chem_comp_id`` is in
     ``structcooker.utils.mapping.CRYSTALLIZATION_AID_CHEMCOMPS`` (the AF3
     COMMON_CRYSTALLIZATION_AIDS list: SO4 GOL EDO PO4 ACT PEG DMS TRS PGE PG4
     FMT EPE MPD MES CD IOD) is dropped from the structure.  Residues go one at a
     time, so a chain disappears only once all of its residues are aids -- the
     same rule ``structcooker...transforms.filtering.filter_ccd`` uses.  This
     rewrites coordinates, so affected entries are re-encoded rather than copied.

  2. *Row* filter -- an edge_node row is kept when it is an interface row
     (``cluster2 != "None"``) whose two cluster types are DNA on one side and a
     protein on the other.  Protein means any of ``cP`` Protein(L), ``cQ``
     Protein(D), ``cA`` Antibody; DNA means ``cD``.  Cluster-type letters are the
     second character of the cluster id -- see ``structcooker.utils.mapping``
     (``cluster_maps``).

  3. *Entry* filter -- after aid removal the structure must be a genuine
     protein-DNA complex: it must hold at least one protein chain AND at least
     one DNA chain, and no RNA (``cR``) or DNA/RNA hybrid (``cN``) chain.  A
     sub-entry that is DNA-only (a bare duplex from another assembly of the same
     PDB entry) or protein-only is dropped even though it breaks no other rule.
     Remaining small molecules are kept as they are: non-aid non-polymer (``cL``),
     branched glycans (``cB``) and unknown (``cX``).

The entry filter runs per ``{assembly}_{model}_{altloc}`` sub-entry, which is the
granularity a row points at and the granularity ``miniworld.data.io.load_cifmol``
looks up.  A PDB entry reaches the output when at least one of its sub-entries
survives, and the blob written for it holds only those sub-entries.

An entry that needs no change at all -- no aid residue anywhere, every sub-entry
kept -- is copied as raw bytes and stays byte-identical to the source.  Anything
else is rebuilt through ``CIFMolAttached.residues[mask].extract()`` and re-encoded
with ``biomol.core.utils.to_bytes``, the same codec the source was written with.

Outputs
-------
  <basedir>/cif_attached_train_onlyPrtDna.lmdb
  <basedir>/cif_attached_valid_2_onlyPrtDna.lmdb
  <basedir>/metadata/train_edge_node_onlyPrtDna.tsv
  <basedir>/metadata/valid2_edge_node_onlyPrtDna.tsv

Usage
-----
Needs biomol + structcooker, so run it with StructCooker's env::

    /home/bsoohyuncd/software/StructCooker/.pixi/envs/default/bin/python \
        scripts/make_prt_dna_set.py --dry-run
    ... /bin/python scripts/make_prt_dna_set.py --split valid2
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes, to_bytes
from structcooker.mols import CIFMolAttached
from structcooker.utils.mapping import CRYSTALLIZATION_AID_CHEMCOMPS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASEDIR = Path("/public_data/bsoohyuncd/BioMolDB_20260224")

SPLITS: dict[str, dict[str, Path]] = {
    "train": {
        "in_lmdb": BASEDIR / "cif_attached_train.lmdb",
        "in_tsv": BASEDIR / "metadata/train_edge_node.tsv",
        "out_lmdb": BASEDIR / "cif_attached_train_onlyPrtDna.lmdb",
        "out_tsv": BASEDIR / "metadata/train_edge_node_onlyPrtDna.tsv",
    },
    "valid2": {
        "in_lmdb": BASEDIR / "cif_attached_valid_2.lmdb",
        "in_tsv": BASEDIR / "metadata/valid2_edge_node.tsv",
        "out_lmdb": BASEDIR / "cif_attached_valid_2_onlyPrtDna.lmdb",
        "out_tsv": BASEDIR / "metadata/valid2_edge_node_onlyPrtDna.tsv",
    },
}

TSV_HEADER = (
    "cluster1\tcluster2\tpdb_id\tassembly_id\tmodel_id\talt_id\tchain_id1\tchain_id2\n"
)

PROTEIN_TYPES = frozenset("PQA")  # Protein(L), Protein(D), Antibody
DNA_TYPE = "D"
FORBIDDEN_TYPES = frozenset("RN")  # RNA, DNA/RNA hybrid
AID_CODES = sorted(CRYSTALLIZATION_AID_CHEMCOMPS)
ZSTD_LEVEL = 6
COMMIT_EVERY = 500


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def is_protein_dna_pair(cluster1: str, cluster2: str) -> bool:
    """True when the two cluster ids are a DNA chain against a protein chain."""
    pair = {cluster1[1], cluster2[1]}
    return DNA_TYPE in pair and bool(pair & PROTEIN_TYPES)


def strip_aids(mol: CIFMolAttached) -> tuple[CIFMolAttached | None, int]:
    """Drop crystallisation-aid residues; return (mol, n_residues_removed).

    ``None`` when every residue was an aid, which leaves nothing to keep.
    """
    ccd = np.asarray(mol.residues.chem_comp_id.value)
    aid = np.isin(ccd, AID_CODES)
    n_aid = int(aid.sum())
    if n_aid == 0:
        return mol, 0
    if n_aid == len(ccd):
        return None, n_aid
    return mol.residues[~aid].extract(), n_aid


def classify(mol: CIFMolAttached) -> tuple[str, set[str]]:
    """Return (verdict, cluster-type letters) for one aid-stripped sub-entry."""
    types = {str(c)[1] for c in np.asarray(mol.chains.cluster_id.value)}
    if types & FORBIDDEN_TYPES:
        return "has_rna", types
    if not types & PROTEIN_TYPES:
        return "no_protein", types
    if DNA_TYPE not in types:
        return "no_dna", types
    return "keep", types


# ---------------------------------------------------------------------------
# Pass 1 -- candidate rows straight out of the TSV
# ---------------------------------------------------------------------------


def read_candidate_rows(tsv_path: Path) -> tuple[dict[str, list[tuple[str, ...]]], Counter]:
    """Group protein-DNA interface rows by lowercase pdb_id."""
    rows_by_pdb: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    tally: Counter[str] = Counter()

    with tsv_path.open() as fh:
        header = next(fh).rstrip("\n")
        if header != TSV_HEADER.rstrip("\n"):
            msg = f"unexpected header in {tsv_path}: {header!r}"
            raise ValueError(msg)
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            tally["rows_total"] += 1
            fields = tuple(stripped.split("\t"))
            if len(fields) != 8:
                msg = f"expected 8 columns, got {len(fields)}: {stripped!r}"
                raise ValueError(msg)
            if fields[1] == "None":
                tally["rows_monomer"] += 1
                continue
            if not is_protein_dna_pair(fields[0], fields[1]):
                continue
            tally["rows_candidate"] += 1
            tally["pair_" + "-".join(sorted({fields[0][1], fields[1][1]}))] += 1
            rows_by_pdb[fields[2].lower()].append(fields)

    tally["pdb_candidate"] = len(rows_by_pdb)
    return rows_by_pdb, tally


# ---------------------------------------------------------------------------
# Pass 2 -- rebuild each candidate entry
# ---------------------------------------------------------------------------


def build_entry(
    raw: bytes,
    rows: list[tuple[str, ...]],
    counts: Counter,
) -> tuple[bytes | None, list[tuple[str, ...]]]:
    """Filter one PDB entry; return (blob or None, surviving rows)."""
    entry = load_bytes(raw)
    kept: dict[str, dict] = {}
    chains_by_key: dict[str, set[str]] = {}
    modified = False

    for asm_key, item in entry.items():
        mol = CIFMolAttached.from_dict(item["cifmol_attached_dict"])
        n_chains_before = len(mol.chains)

        stripped, n_aid = strip_aids(mol)
        if n_aid:
            modified = True
            counts["aid_residues_removed"] += n_aid
        if stripped is None:
            counts["subentry_all_aid"] += 1
            continue
        if n_aid:
            counts["aid_chains_removed"] += n_chains_before - len(stripped.chains)

        verdict, _ = classify(stripped)
        counts[f"subentry_{verdict}"] += 1
        if verdict != "keep":
            modified = True
            continue

        new_item = dict(item)
        new_item["cifmol_attached_dict"] = stripped.to_dict()
        kept[asm_key] = new_item
        chains_by_key[asm_key] = {str(c) for c in np.asarray(stripped.chains.chain_id.value)}

    surviving: list[tuple[str, ...]] = []
    for row in rows:
        asm_key = f"{row[3]}_{row[4]}_{row[5]}"
        if asm_key not in kept:
            counts["row_dropped_subentry"] += 1
            continue
        if row[6] not in chains_by_key[asm_key] or row[7] not in chains_by_key[asm_key]:
            counts["row_dropped_chain_absent"] += 1
            continue
        surviving.append(row)

    if not surviving:
        return None, []
    if modified:
        counts["entry_reencoded"] += 1
        return to_bytes(kept, level=ZSTD_LEVEL), surviving
    counts["entry_verbatim"] += 1
    return raw, surviving


def process_split(name: str, paths: dict[str, Path], *, dry_run: bool) -> None:  # noqa: PLR0915
    """Filter one split and, unless dry-running, write its LMDB + TSV."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"source lmdb : {paths['in_lmdb']}")
    print(f"source tsv  : {paths['in_tsv']}")
    print(f"aid codes   : {' '.join(AID_CODES)}")

    rows_by_pdb, tally = read_candidate_rows(paths["in_tsv"])
    print(f"\nrows in tsv                     : {tally['rows_total']}")
    print(f"  monomer rows (cluster2 None)  : {tally['rows_monomer']}")
    print(f"protein-DNA interface rows      : {tally['rows_candidate']}")
    for key in sorted(k for k in tally if k.startswith("pair_")):
        print(f"  {key[5:]:28s}: {tally[key]}")
    print(f"candidate pdb entries           : {tally['pdb_candidate']}")

    env = lmdb.open(str(paths["in_lmdb"]), readonly=True, lock=False, max_readers=2048)
    print(f"source entries in lmdb          : {env.stat()['entries']}")

    kept_rows: list[tuple[str, ...]] = []
    out_values: dict[bytes, bytes] = {}
    counts: Counter[str] = Counter()
    started = time.time()

    with env.begin(buffers=True) as txn:
        for n_done, pdb in enumerate(sorted(rows_by_pdb), start=1):
            raw = txn.get(pdb.encode())
            if raw is None:
                counts["pdb_missing_from_lmdb"] += 1
                continue
            blob, surviving = build_entry(bytes(raw), rows_by_pdb[pdb], counts)
            if blob is None:
                counts["pdb_dropped"] += 1
            else:
                out_values[pdb.encode()] = blob
                kept_rows.extend(surviving)
                counts["pdb_kept"] += 1
            if n_done % COMMIT_EVERY == 0:
                print(
                    f"  scanned {n_done}/{len(rows_by_pdb)} "
                    f"({time.time() - started:.0f}s, kept {counts['pdb_kept']})",
                    flush=True,
                )

    env.close()

    payload = sum(len(v) for v in out_values.values())
    print(f"\n-- aid removal ({time.time() - started:.0f}s)")
    print(f"aid residues removed              : {counts['aid_residues_removed']}")
    print(f"chains emptied by aid removal     : {counts['aid_chains_removed']}")
    print("\n-- entry filter")
    print(f"sub-entries kept                  : {counts['subentry_keep']}")
    print(f"sub-entries dropped, RNA/hybrid   : {counts['subentry_has_rna']}")
    print(f"sub-entries dropped, no protein   : {counts['subentry_no_protein']}")
    print(f"sub-entries dropped, no DNA       : {counts['subentry_no_dna']}")
    if counts["subentry_all_aid"]:
        print(f"sub-entries dropped, all aid      : {counts['subentry_all_aid']}")
    print(f"pdb entries kept                  : {counts['pdb_kept']}")
    print(f"  copied verbatim                 : {counts['entry_verbatim']}")
    print(f"  re-encoded                      : {counts['entry_reencoded']}")
    print(f"pdb entries dropped               : {counts['pdb_dropped']}")
    if counts["pdb_missing_from_lmdb"]:
        print(f"pdb entries missing from lmdb     : {counts['pdb_missing_from_lmdb']}")
    print(f"rows kept                         : {len(kept_rows)}")
    print(f"  dropped, sub-entry gone         : {counts['row_dropped_subentry']}")
    if counts["row_dropped_chain_absent"]:
        print(f"  dropped, chain absent           : {counts['row_dropped_chain_absent']}")
    print(f"output payload                    : {payload / 1e9:.3f} GB")

    if dry_run:
        print("\n[dry-run] nothing written.")
        return

    write_tsv(paths["out_tsv"], kept_rows)
    write_lmdb(paths["out_lmdb"], out_values, payload)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    """Write the filtered edge_node TSV, sorted like the source format."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write(TSV_HEADER)
        for row in sorted(rows, key=lambda r: (r[2], r[3], r[4], r[5], r[6], r[7])):
            fh.write("\t".join(row) + "\n")
    tmp.replace(path)
    print(f"\nwrote {len(rows)} rows -> {path}")


def write_lmdb(path: Path, values: dict[bytes, bytes], payload: int) -> None:
    """Write the filtered LMDB, sized to what it actually holds."""
    if path.exists():
        if not path.name.endswith(".lmdb"):
            msg = f"refusing to delete non-.lmdb path: {path}"
            raise ValueError(msg)
        print(f"removing existing {path}")
        shutil.rmtree(path)

    map_size = int(payload * 1.5) + 256 * 1024 * 1024
    env = lmdb.open(str(path), map_size=map_size)
    pending: list[tuple[bytes, bytes]] = []
    for key in sorted(values):
        pending.append((key, values[key]))
        if len(pending) >= 2000:
            with env.begin(write=True) as txn:
                for k, v in pending:
                    txn.put(k, v)
            pending.clear()
    if pending:
        with env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k, v)
    env.sync()
    written = env.stat()["entries"]
    env.close()

    if written != len(values):
        msg = f"wrote {len(values)} keys but the new env holds {written}"
        raise SystemExit(msg)
    print(
        f"wrote {written} keys -> {path} "
        f"({payload / 1e9:.3f} GB payload, map_size={map_size / 1e9:.2f} GB)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse the command line and run the requested splits."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    names = list(SPLITS) if args.split == "all" else [args.split]
    for name in names:
        process_split(name, SPLITS[name], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
