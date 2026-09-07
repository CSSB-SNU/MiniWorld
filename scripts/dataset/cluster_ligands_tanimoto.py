#!/usr/bin/env python3
"""Cluster every ligand by ECFP4 Tanimoto, replacing the one-to-one cluster ids.

Why
---
``metadata/seq_cluster30.tsv`` clusters polymers at 30% sequence identity -- 48,903
protein clusters over ~97k sequences, mean 1.98 members, max 922 -- but its 45,637
``cL`` rows all have exactly one member.  Sequence identity is undefined for a
small molecule, so ligands were given cluster ids purely to fit the
``cluster1``/``cluster2`` schema; ``cL0000004`` means "magnesium", not "the
magnesium-like cluster".

The practical cost is in sampling.  ``dataloader._load_items`` groups rows by
``edge_id = f"{cluster1}_{cluster2}"`` and gives every edge_id equal mass within
its type, so a congeneric series of 200 analogues draws 200x the sampling weight
of a singleton ligand, purely because 200 analogues were crystallised.

This script gives the ligand axis a real clustering, so both sides of an
interface are balanced at cluster granularity.

Method
------
Morgan fingerprints (ECFP4: radius 2, 4096 bits) from the CCD store's
``rdkit_smiles``, then greedy sphere exclusion at ``--threshold`` Tanimoto: visit
ligands in order of how often they appear in the edge_node table, most first;
each unvisited ligand opens a cluster and absorbs every later ligand within the
threshold.  The most-used ligand therefore names its cluster, and no two cluster
representatives are more than ``--threshold`` similar.

A ligand whose SMILES is missing or unparseable, and every single-atom ion, forms
its own singleton cluster -- an ion sets one Morgan bit and scores Tanimoto 0
against every other ion, so it could never merge anyway.

Outputs
-------
``--out-clusters`` (default ``metadata/ligand_cluster_tanimoto50.tsv``)
    Same shape as ``seq_cluster30.tsv``: ``<cluster_id>\\t<seq_id,seq_id,...>``,
    the cluster id being the representative's own ``cL`` id.  Drop-in for any
    code that already reads the sequence-cluster format.

``--out-map`` (default ``metadata/ligand_cluster_map_tanimoto50.tsv``)
    ``<old_cluster_id>\\t<new_cluster_id>`` for every ligand, including ligands
    that are their own representative.  This is the lookup the dataloader needs
    to switch ``cluster2`` when building ``edge_id``.

Usage
-----
    /home/bsoohyuncd/.conda/envs/prolif/bin/python \\
        scripts/dataset/cluster_ligands_tanimoto.py [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

BASEDIR = Path("/public_data/bsoohyuncd/BioMolDB_20260224")
SEQ_ID_MAP = BASEDIR / "metadata/seq_id_map.tsv"
EDGE_TSV = BASEDIR / "metadata/train_edge_node.tsv"
CCD_LMDB = Path("/public_data/bsoohyuncd/CCD/preprocessed_CCD_20260826.lmdb")
OUT_CLUSTERS = BASEDIR / "metadata/ligand_cluster_tanimoto50.tsv"
OUT_MAP = BASEDIR / "metadata/ligand_cluster_map_tanimoto50.tsv"

CCD_TOKEN = re.compile(r"\(([^)]*)\)")


def load_bytes(byte_data: bytes) -> dict:
    """Decode one BioMolDB blob.

    A standalone copy of ``biomol.core.utils.load_bytes`` -- identical format,
    but this script runs in the RDKit environment, which has no biomol.  Only
    zstandard, numpy and json are needed to read it.
    """
    import json
    from io import BytesIO

    import numpy as np
    from zstandard import ZstdDecompressor

    with ZstdDecompressor().stream_reader(BytesIO(byte_data)) as reader:
        raw = reader.read()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    payload = memoryview(raw)[8 + header_len :]

    flat = {}
    offset = 0
    for key, length in header["arrays"].items():
        flat[key] = payload[offset : offset + length]
        offset += length

    def rebuild(template: dict) -> dict:
        out = {}
        for key, value in template.items():
            if isinstance(value, str) and value in flat:
                out[key] = np.load(BytesIO(flat[value]), allow_pickle=False)
            elif isinstance(value, dict):
                out[key] = rebuild(value)
            else:
                out[key] = value
        return out

    return rebuild(header["template"])


def load_ligand_seq_ids(path: Path) -> dict[str, list[str]]:
    """ligand seq_id -> its CCD component codes."""
    out: dict[str, list[str]] = {}
    with path.open() as fh:
        for line in fh:
            seq_id, _, seq = line.rstrip("\n").partition("\t")
            if seq_id[:1] == "L":
                out[seq_id] = CCD_TOKEN.findall(seq.split("|")[0])
    return out


def count_usage(path: Path) -> Counter:
    """How often each ligand cluster appears in the edge table, any edge type."""
    usage: Counter[str] = Counter()
    with path.open() as fh:
        next(fh)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                continue
            for cluster in (fields[0], fields[1]):
                if cluster[:2] == "cL":
                    usage[cluster] += 1
    return usage


def main() -> None:  # noqa: PLR0915
    """Cluster the ligands and write both output tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Tanimoto above which two ligands share a cluster (default 0.5).")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (default 2 = ECFP4).")
    ap.add_argument("--bits", type=int, default=4096, help="Fingerprint size (default 4096).")
    ap.add_argument("--ccd-lmdb", type=Path, default=CCD_LMDB)
    ap.add_argument("--out-clusters", type=Path, default=OUT_CLUSTERS)
    ap.add_argument("--out-map", type=Path, default=OUT_MAP)
    args = ap.parse_args()

    import lmdb
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")

    print(f"ccd store : {args.ccd_lmdb}")
    print(f"threshold : Tanimoto > {args.threshold} | ECFP r={args.radius} bits={args.bits}")

    seq2ccd = load_ligand_seq_ids(SEQ_ID_MAP)
    print(f"\nligand seq_ids            : {len(seq2ccd):,}")
    usage = count_usage(EDGE_TSV)
    print(f"ligand clusters used in train edge table : {len(usage):,}")

    # ---- fingerprints ---------------------------------------------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.bits)
    env = lmdb.open(str(args.ccd_lmdb), readonly=True, lock=False)
    fps: dict[str, object] = {}
    singleton: list[str] = []
    with env.begin() as txn:
        for seq_id, comps in seq2ccd.items():
            smiles = None
            if len(comps) == 1:
                raw = txn.get(comps[0].encode())
                if raw is not None:
                    nodes = load_bytes(bytes(raw))["residues"]["nodes"]
                    if "rdkit_smiles" in nodes:
                        smiles = str(np.asarray(nodes["rdkit_smiles"]["value"]).reshape(-1)[0])
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None or mol.GetNumHeavyAtoms() < 2:  # noqa: PLR2004
                singleton.append(seq_id)   # ions and unparseables cluster alone
            else:
                fps[seq_id] = gen.GetFingerprint(mol)
    env.close()
    print(f"fingerprinted             : {len(fps):,}")
    print(f"forced singletons (ion / no smiles) : {len(singleton):,}")

    # ---- greedy sphere exclusion, most-used ligand first ------------------
    order = sorted(fps, key=lambda s: (-usage.get("c" + s, 0), s))
    n = len(order)
    print(f"\nclustering {n:,} ligands ...", flush=True)
    started = time.time()
    fp_list = [fps[s] for s in order]
    assigned: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    alive = [True] * n
    for i in range(n):
        if not alive[i]:
            continue
        rep = order[i]
        assigned[rep] = rep
        members[rep] = [rep]
        rest = [j for j in range(i + 1, n) if alive[j]]
        if rest:
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], [fp_list[j] for j in rest])
            for j, sim in zip(rest, sims):
                if sim > args.threshold:
                    alive[j] = False
                    assigned[order[j]] = rep
                    members[rep].append(order[j])
        if (i + 1) % 5000 == 0:
            print(f"  visited {i + 1:,}/{n:,} (clusters {len(members):,}) "
                  f"({time.time() - started:.0f}s)", flush=True)

    for seq_id in singleton:
        assigned[seq_id] = seq_id
        members[seq_id] = [seq_id]

    sizes = Counter(len(v) for v in members.values())
    biggest = max(members.items(), key=lambda kv: len(kv[1]))
    print(f"\n-- results ({time.time() - started:.0f}s)")
    print(f"ligands clustered  : {len(assigned):,}")
    print(f"clusters produced  : {len(members):,}")
    print(f"mean cluster size  : {len(assigned) / len(members):.2f}")
    print(f"largest cluster    : c{biggest[0]} with {len(biggest[1])} ligands")
    print(f"size distribution  : " + ", ".join(
        f"{size}:{count}" for size, count in sorted(sizes.items())[:8]) + " ...")
    print(f"singleton clusters : {sizes[1]:,} "
          f"({100 * sizes[1] / len(members):.1f}% of clusters)")

    # ---- write ------------------------------------------------------------
    args.out_clusters.parent.mkdir(parents=True, exist_ok=True)
    with args.out_clusters.open("w") as fh:
        for rep in sorted(members):
            fh.write(f"c{rep}\t{','.join(sorted(members[rep]))}\n")
    print(f"\nwrote {len(members):,} clusters -> {args.out_clusters}")

    with args.out_map.open("w") as fh:
        fh.write("old_cluster\tnew_cluster\n")
        for seq_id in sorted(assigned):
            fh.write(f"c{seq_id}\tc{assigned[seq_id]}\n")
    print(f"wrote {len(assigned):,} mappings -> {args.out_map}")

    moved = sum(1 for s, r in assigned.items() if s != r)
    print(f"\n{moved:,} of {len(assigned):,} ligands move to another ligand's cluster id "
          f"({100 * moved / len(assigned):.1f}%)")


if __name__ == "__main__":
    main()
