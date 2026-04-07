#!/usr/bin/env python3
"""
Create a validation set from cif_attached_valid_1.lmdb and cif_attached_valid_2.lmdb.

Criteria (OR condition):
1. At least 10% of residues in total length are modified (non-standard residues in polymer chains)
2. Contains glycan residues (sugar molecules)
"""

import lmdb
import zstandard as zstd
import json
import struct
import numpy as np
import io
import os
from collections import defaultdict

# Standard amino acids
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}

# Standard nucleotides (DNA + RNA)
STANDARD_NT = {
    'DA', 'DC', 'DG', 'DT', 'DU',   # DNA
    'A', 'C', 'G', 'U', 'I',         # RNA
}

STANDARD_RESIDUES = STANDARD_AA | STANDARD_NT

# Common glycan/sugar residue names (from CCD)
GLYCAN_RESIDUES = {
    'NAG', 'NDG', 'BMA', 'MAN', 'GAL', 'GLC', 'FUC', 'SIA', 'BGC',
    'GLA', 'NGA', 'A2G', 'FUL', 'RAM', 'XYS', 'RIB', 'ARA',
    'GCS', 'GCU', 'IDR', 'SGN', 'SUC', 'TRE', 'XYP', 'LMT',
    'G6D', 'GCW', 'MAV', 'AFD', 'ALL', 'SHD', 'BM3', 'BM7',
    'BOG', 'GL0', 'AHR', 'DDA', 'DDL', 'MDA', 'MUB',
}

# Polymer entity types (polypeptide + polynucleotide)
POLYMER_ENTITY_TYPES = {'polypeptide(L)', 'polypeptide(D)', 'polyribonucleotide', 'polydeoxyribonucleotide'}

BASEDIR = "/public_data/bsoohyuncd/BioMolDB_20260224"
LMDB_FILES = [
    os.path.join(BASEDIR, "cif_attached_valid_1.lmdb"),
    os.path.join(BASEDIR, "cif_attached_valid_2.lmdb"),
]
OUTPUT_LMDB = os.path.join(BASEDIR, "cif_attached_valid_modified_glycan.lmdb")


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

    # Load all numpy arrays sequentially
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

    # Build UUID -> array mapping by matching sizes
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


def check_entry(template, arrays):
    """
    Check if an entry meets criteria (OR condition):
    1. At least 10% modified residues in total polymer chain length
    2. Contains glycan residues or branched entity types

    Returns: (meets_criteria, stats_dict)
    """
    total_polymer_residues = 0
    total_modified_residues = 0
    has_glycan = False
    has_branched = False
    modified_types = set()
    glycan_types = set()

    for asm_key, asm_data in template['template'].items():
        chain_data = asm_data['cifmol_attached_dict']

        et_uuid = chain_data['chains']['nodes']['entity_type']['value']
        entity_types = arrays.get(et_uuid)
        if entity_types is None:
            continue

        cc_uuid = chain_data['residues']['nodes']['chem_comp_id']['value']
        chem_comps = arrays.get(cc_uuid)
        if chem_comps is None:
            continue

        chain_res_indptr = chain_data['index_table']['chain_res_indptr']

        # Check for branched entity type (glycan chains)
        for et in entity_types:
            if 'branched' in str(et).lower():
                has_branched = True

        for ci in range(len(entity_types)):
            start = chain_res_indptr[ci]
            end = chain_res_indptr[ci + 1]
            chain_ccs = chem_comps[start:end]
            et = str(entity_types[ci])

            # Check polymer chains for modified residues
            if any(pet in et for pet in POLYMER_ENTITY_TYPES):
                n_chain = len(chain_ccs)
                modified_mask = np.array([cc not in STANDARD_RESIDUES for cc in chain_ccs])
                n_modified = int(modified_mask.sum())
                total_polymer_residues += n_chain
                total_modified_residues += n_modified
                if n_modified > 0:
                    modified_types.update(chain_ccs[modified_mask].tolist())

            # Check all chains for glycan residues
            for cc in chain_ccs:
                if cc in GLYCAN_RESIDUES:
                    has_glycan = True
                    glycan_types.add(cc)

        break  # Only check first assembly to avoid double-counting

    mod_ratio = total_modified_residues / total_polymer_residues if total_polymer_residues > 0 else 0
    meets_modified = mod_ratio >= 0.10
    meets_glycan = has_glycan or has_branched

    stats = {
        'total_polymer_residues': total_polymer_residues,
        'total_modified_residues': total_modified_residues,
        'mod_ratio': mod_ratio,
        'has_glycan': has_glycan,
        'has_branched': has_branched,
        'modified_types': modified_types,
        'glycan_types': glycan_types,
        'reason': [],
    }
    if meets_modified:
        stats['reason'].append('modified>=10%')
    if meets_glycan:
        stats['reason'].append('glycan')

    return meets_modified or meets_glycan, stats


def main():
    # Phase 1: Scan both LMDBs and collect qualifying keys
    qualifying = []  # List of (lmdb_path, key, stats)

    for lmdb_path in LMDB_FILES:
        print(f"\nScanning {os.path.basename(lmdb_path)}...")
        env = lmdb.open(lmdb_path, readonly=True, lock=False)

        with env.begin() as txn:
            n_entries = txn.stat()['entries']
            print(f"  Total entries: {n_entries}")

            cursor = txn.cursor()
            count = 0
            n_modified_only = 0
            n_glycan_only = 0
            n_both = 0
            n_either = 0

            for key, _ in cursor:
                count += 1
                if count % 500 == 0:
                    print(f"  Processed {count}/{n_entries}...")

                try:
                    template, arrays = load_entry(txn, key)
                    if template is None:
                        continue
                    meets, stats = check_entry(template, arrays)

                    has_mod = stats['mod_ratio'] >= 0.10
                    has_gly = stats['has_glycan'] or stats['has_branched']

                    if has_mod:
                        n_modified_only += 1
                    if has_gly:
                        n_glycan_only += 1
                    if has_mod and has_gly:
                        n_both += 1
                    if meets:
                        n_either += 1
                        qualifying.append((lmdb_path, key, stats))

                except Exception as e:
                    print(f"  Error processing {key}: {e}")

            print(f"  Scanned: {count}")
            print(f"  >= 10% modified: {n_modified_only}")
            print(f"  Has glycan: {n_glycan_only}")
            print(f"  Both: {n_both}")
            print(f"  Either (qualifying): {n_either}")

        env.close()

    print(f"\n{'='*60}")
    print(f"Total qualifying entries: {len(qualifying)}")

    if len(qualifying) == 0:
        print("No entries meet the criteria. Consider relaxing thresholds.")
        return

    # Phase 2: Copy qualifying entries to new LMDB
    print(f"\nWriting qualifying entries to {os.path.basename(OUTPUT_LMDB)}...")

    # Estimate map size
    total_size = 0
    src_envs = {}
    for lmdb_path, key, stats in qualifying:
        if lmdb_path not in src_envs:
            src_envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False)
        env = src_envs[lmdb_path]
        with env.begin() as txn:
            val = txn.get(key)
            if val:
                total_size += len(val)

    map_size = int(total_size * 1.5) + 10 * 1024 * 1024  # 1.5x + 10MB headroom

    out_env = lmdb.open(OUTPUT_LMDB, map_size=map_size)

    written = 0
    for lmdb_path, key, stats in qualifying:
        env = src_envs[lmdb_path]
        with env.begin() as txn:
            val = txn.get(key)
            if val:
                with out_env.begin(write=True) as out_txn:
                    out_txn.put(key, val)
                    written += 1

    out_env.close()
    for env in src_envs.values():
        env.close()

    print(f"Written {written} entries to {OUTPUT_LMDB}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total qualifying entries: {len(qualifying)}")

    # Count by reason
    mod_count = sum(1 for _, _, s in qualifying if 'modified>=10%' in s['reason'])
    gly_count = sum(1 for _, _, s in qualifying if 'glycan' in s['reason'])
    both_count = sum(1 for _, _, s in qualifying if len(s['reason']) == 2)
    print(f"  Modified only: {mod_count - both_count}")
    print(f"  Glycan only: {gly_count - both_count}")
    print(f"  Both: {both_count}")

    print(f"\nPer-entry details:")
    for lmdb_path, key, stats in qualifying:
        src = os.path.basename(lmdb_path)
        reason = '+'.join(stats['reason'])
        print(f"  {key.decode():10s} ({src}): "
              f"mod={stats['total_modified_residues']}/{stats['total_polymer_residues']} "
              f"({stats['mod_ratio']:.1%}), "
              f"glycan={stats['glycan_types'] or 'none'}, "
              f"branched={'Y' if stats['has_branched'] else 'N'}, "
              f"reason={reason}")


if __name__ == '__main__':
    main()
