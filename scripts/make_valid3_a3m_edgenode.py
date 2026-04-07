#!/usr/bin/env python3
"""
Generate a3m.lmdb and valid3_edge_node.tsv for cif_attached_valid_modified_glycan.lmdb.

1. a3m.lmdb: Copy MSA entries from the main a3m.lmdb for all seq_ids referenced
   by chains in the qualifying LMDB.
2. valid3_edge_node.tsv: Build cluster_id -> chain reference mapping in the same
   format as valid1_edge_node.tsv / valid2_edge_node.tsv.
"""

import lmdb
import zstandard as zstd
import json
import struct
import numpy as np
import io
import os
from collections import defaultdict

BASEDIR = "/public_data/bsoohyuncd/BioMolDB_20260224"
INPUT_LMDB = os.path.join(BASEDIR, "cif_attached_valid_modified_glycan.lmdb")
SRC_A3M_LMDB = os.path.join(BASEDIR, "a3m.lmdb")
OUT_A3M_LMDB = os.path.join(BASEDIR, "valid3_a3m.lmdb")
OUT_EDGE_NODE = os.path.join(BASEDIR, "metadata", "valid3_edge_node.tsv")


def load_entry(txn, key_bytes):
    """Load and parse a single LMDB entry."""
    val = txn.get(key_bytes)
    if val is None:
        return None, None

    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(val)
    header_len = struct.unpack('<Q', decompressed[:8])[0]
    template = json.loads(decompressed[8:8 + header_len])
    array_data = decompressed[8 + header_len:]

    buf = io.BytesIO(array_data)
    loaded = []
    try:
        while True:
            pos = buf.tell()
            arr = np.load(buf, allow_pickle=False)
            end = buf.tell()
            loaded.append((end - pos, arr))
    except Exception:
        pass

    size_to_arrays = {}
    for size, arr in loaded:
        size_to_arrays.setdefault(size, []).append(arr)

    uuid_by_size = {}
    for uuid, size in template['arrays'].items():
        uuid_by_size.setdefault(size, []).append(uuid)

    uuid_to_array = {}
    for size, uuids in uuid_by_size.items():
        arrs = size_to_arrays.get(size, [])
        for i, uuid in enumerate(uuids):
            if i < len(arrs):
                uuid_to_array[uuid] = arrs[i]

    return template, uuid_to_array


def main():
    # Phase 1: Scan qualifying LMDB to collect seq_ids and build edge_node data
    print("Phase 1: Scanning qualifying LMDB for seq_ids and chain info...")

    all_seq_ids = set()
    # cluster_id -> set of chain references: "{PDB_upper}_{asm}_{model}_{alt}_({chain_id})"
    cluster_to_refs = defaultdict(set)

    env = lmdb.open(INPUT_LMDB, readonly=True, lock=False)
    with env.begin() as txn:
        n_entries = txn.stat()['entries']
        print(f"  Total entries: {n_entries}")

        cursor = txn.cursor()
        count = 0
        for key, _ in cursor:
            count += 1
            if count % 500 == 0:
                print(f"  Processed {count}/{n_entries}...")

            pdb_code = key.decode()
            pdb_upper = pdb_code.upper()

            try:
                template, arrays = load_entry(txn, key)
                if template is None:
                    continue

                for asm_key, asm_data in template['template'].items():
                    chain_data = asm_data['cifmol_attached_dict']
                    chains = chain_data['chains']['nodes']

                    et_uuid = chains['entity_type']['value']
                    entity_types = arrays.get(et_uuid)
                    if entity_types is None:
                        continue

                    seq_uuid = chains['seq_id']['value']
                    seq_ids = arrays.get(seq_uuid)

                    cluster_uuid = chains['cluster_id']['value']
                    cluster_ids = arrays.get(cluster_uuid)

                    chain_id_uuid = chains['chain_id']['value']
                    chain_ids = arrays.get(chain_id_uuid)

                    if seq_ids is None or cluster_ids is None or chain_ids is None:
                        continue

                    for i in range(len(entity_types)):
                        seq_id = str(seq_ids[i])
                        cluster_id = str(cluster_ids[i])
                        chain_id = str(chain_ids[i])

                        all_seq_ids.add(seq_id)

                        # Build chain reference: PDB_asm_key_(chain_id)
                        ref = f"{pdb_upper}_{asm_key}_({chain_id})"
                        cluster_to_refs[cluster_id].add(ref)

            except Exception as e:
                print(f"  Error processing {pdb_code}: {e}")

    env.close()

    print(f"  Unique seq_ids: {len(all_seq_ids)}")
    print(f"  Unique cluster_ids: {len(cluster_to_refs)}")

    # Phase 2: Write valid3_edge_node.tsv
    print(f"\nPhase 2: Writing {os.path.basename(OUT_EDGE_NODE)}...")

    with open(OUT_EDGE_NODE, 'w') as f:
        for cluster_id in sorted(cluster_to_refs.keys()):
            refs = sorted(cluster_to_refs[cluster_id])
            refs_str = ",".join(refs)
            f.write(f"{cluster_id}\tNone\t{refs_str}\n")

    n_lines = len(cluster_to_refs)
    print(f"  Written {n_lines} lines")

    # Phase 3: Copy matching entries from a3m.lmdb
    print(f"\nPhase 3: Copying matching a3m entries...")

    src_env = lmdb.open(SRC_A3M_LMDB, readonly=True, lock=False)

    # First pass: check which seq_ids exist in a3m.lmdb and estimate size
    found_keys = []
    total_size = 0
    missing = []

    with src_env.begin() as txn:
        for seq_id in sorted(all_seq_ids):
            val = txn.get(seq_id.encode())
            if val is not None:
                found_keys.append(seq_id)
                total_size += len(val)
            else:
                missing.append(seq_id)

    print(f"  Found {len(found_keys)}/{len(all_seq_ids)} seq_ids in a3m.lmdb")
    if missing:
        # Show breakdown by prefix
        miss_prefix = defaultdict(int)
        for m in missing:
            miss_prefix[m[0]] = miss_prefix.get(m[0], 0) + 1
        print(f"  Missing by prefix: {dict(miss_prefix)}")

    # Second pass: copy to new LMDB
    map_size = int(total_size * 1.5) + 10 * 1024 * 1024
    out_env = lmdb.open(OUT_A3M_LMDB, map_size=map_size)

    written = 0
    with src_env.begin() as src_txn:
        for seq_id in found_keys:
            val = src_txn.get(seq_id.encode())
            if val:
                with out_env.begin(write=True) as out_txn:
                    out_txn.put(seq_id.encode(), val)
                    written += 1

            if written % 500 == 0 and written > 0:
                print(f"  Written {written}/{len(found_keys)}...")

    out_env.close()
    src_env.close()

    print(f"  Written {written} entries to {os.path.basename(OUT_A3M_LMDB)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {os.path.basename(INPUT_LMDB)} ({n_entries} entries)")
    print(f"Output a3m: {os.path.basename(OUT_A3M_LMDB)} ({written} entries)")
    print(f"Output edge_node: {os.path.basename(OUT_EDGE_NODE)} ({n_lines} lines)")
    print(f"Unique seq_ids collected: {len(all_seq_ids)}")
    print(f"  Found in a3m.lmdb: {len(found_keys)}")
    print(f"  Missing (non-polymer/branched, no MSA): {len(missing)}")


if __name__ == '__main__':
    main()
