"""
Check that every chem_comp_id in cif_attached_*.lmdb has a matching key
in ../preprocessed_CCD.lmdb. For any missing IDs, fetch canonical SMILES
from PDB REST API and save results to result.py.
"""

import io
import json
import time
import urllib.request
import urllib.error

import lmdb
import numpy as np
import zstandard

BASE_DIR = "/public_data/bsoohyuncd/BioMolDB_20260224"
CCD_PATH = "/public_data/bsoohyuncd/preprocessed_CCD.lmdb"

LMDB_PATHS = [
    ("cif_attached_train", f"{BASE_DIR}/cif_attached_train.lmdb"),
    ("cif_attached_valid_1", f"{BASE_DIR}/cif_attached_valid_1.lmdb"),
    ("cif_attached_valid_2", f"{BASE_DIR}/cif_attached_valid_2.lmdb"),
]


def load_ccd_keys(path):
    env = lmdb.open(path, readonly=True, lock=False)
    keys = set()
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, _ in cursor:
            keys.add(k.decode())
    env.close()
    return keys


def parse_schema(decompressed):
    content = decompressed[8:]
    depth = 0
    json_end = 0
    for i, b in enumerate(content):
        if b == ord("{"):
            depth += 1
        elif b == ord("}"):
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break
    schema = json.loads(content[:json_end])
    binary_data = content[json_end:]
    return schema, binary_data


def extract_chem_comp_ids(schema, binary_data):
    arrays = schema["arrays"]
    template = schema.get("template", {})
    all_ids = set()

    for chain_key, chain_data in template.items():
        cifmol = chain_data.get("cifmol_attached_dict")
        if not cifmol:
            continue
        residues = cifmol.get("residues", {})
        nodes = residues.get("nodes", {})
        chem_comp = nodes.get("chem_comp_id", {})
        if "value" not in chem_comp:
            continue
        uuid = chem_comp["value"]
        if uuid not in arrays:
            continue

        offset = 0
        for u, size in arrays.items():
            if u == uuid:
                break
            offset += size

        buf = io.BytesIO(binary_data[offset : offset + arrays[uuid]])
        arr = np.load(buf)
        all_ids.update(arr.tolist())

    return all_ids


def scan_lmdb(name, path, ccd_keys):
    dctx = zstandard.ZstdDecompressor()
    env = lmdb.open(path, readonly=True, lock=False)

    all_comp_ids = set()
    missing_to_pdbs = {}  # missing_id -> set of pdb keys
    errors = []

    with env.begin() as txn:
        total = txn.stat()["entries"]
        print(f"\n[{name}] Scanning {total} entries...")
        cursor = txn.cursor()
        t0 = time.time()

        for i, (k, v) in enumerate(cursor):
            pdb_id = k.decode()
            try:
                decompressed = dctx.decompress(v)
                schema, binary_data = parse_schema(decompressed)
                ids = extract_chem_comp_ids(schema, binary_data)
                all_comp_ids.update(ids)

                missing = ids - ccd_keys
                for mid in missing:
                    missing_to_pdbs.setdefault(mid, set()).add(pdb_id)
            except Exception as e:
                errors.append((pdb_id, str(e)))

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{name}] {i+1}/{total} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    env.close()

    print(f"  [{name}] Done. {total} entries in {elapsed:.1f}s")
    print(f"  [{name}] Unique chem_comp_ids: {len(all_comp_ids)}")
    missing_ids = all_comp_ids - ccd_keys
    print(f"  [{name}] Missing from CCD: {len(missing_ids)}")
    if errors:
        print(f"  [{name}] Errors: {len(errors)}")
        for pdb_id, err in errors[:5]:
            print(f"    {pdb_id}: {err}")

    return all_comp_ids, missing_to_pdbs


def fetch_smiles(comp_id):
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            descriptors = data.get("rcsb_chem_comp_descriptor", {})
            smiles_list = descriptors.get("smiles", [])
            # Prefer canonical SMILES
            for entry in smiles_list:
                if isinstance(entry, dict):
                    if entry.get("type") == "SMILES_CANONICAL":
                        return entry.get("string", "")
            # Fallback to first SMILES
            if smiles_list:
                if isinstance(smiles_list[0], dict):
                    return smiles_list[0].get("string", "")
                return str(smiles_list[0])
            # Try comp_descriptor field
            comp_desc = data.get("pdbx_chem_comp_descriptor", [])
            for desc in comp_desc:
                if isinstance(desc, dict) and "SMILES" in desc.get("type", ""):
                    return desc.get("descriptor", "")
            return ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"NOT_FOUND_IN_PDB"
        return f"HTTP_ERROR_{e.code}"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("Loading preprocessed_CCD keys...")
    ccd_keys = load_ccd_keys(CCD_PATH)
    print(f"  {len(ccd_keys)} CCD keys loaded.")

    all_missing_to_pdbs = {}
    global_comp_ids = set()

    for name, path in LMDB_PATHS:
        comp_ids, missing_to_pdbs = scan_lmdb(name, path, ccd_keys)
        global_comp_ids.update(comp_ids)
        for mid, pdbs in missing_to_pdbs.items():
            all_missing_to_pdbs.setdefault(mid, {}).setdefault("pdbs", set())
            all_missing_to_pdbs[mid]["pdbs"].update(pdbs)

    all_missing_ids = sorted(all_missing_to_pdbs.keys())
    print(f"\n=== SUMMARY ===")
    print(f"Total unique chem_comp_ids across all LMDBs: {len(global_comp_ids)}")
    print(f"Total missing from preprocessed_CCD: {len(all_missing_ids)}")

    if all_missing_ids:
        print(f"\nFetching canonical SMILES for {len(all_missing_ids)} missing IDs...")
        missing_dict = {}
        for i, mid in enumerate(all_missing_ids):
            smiles = fetch_smiles(mid)
            missing_dict[mid] = smiles
            pdb_count = len(all_missing_to_pdbs[mid]["pdbs"])
            print(f"  [{i+1}/{len(all_missing_ids)}] {mid}: SMILES={smiles[:60] if smiles else 'N/A'} (in {pdb_count} PDB entries)")
            time.sleep(0.1)  # rate limit

        # Save result.py
        result_path = f"{BASE_DIR}/result.py"
        with open(result_path, "w") as f:
            f.write("# Missing chem_comp_ids not found in preprocessed_CCD.lmdb\n")
            f.write("# Keys: chem_comp_id, Values: canonical SMILES from PDB\n\n")
            f.write("missing_chem_comp_ids = {\n")
            for mid in sorted(missing_dict.keys()):
                smiles = missing_dict[mid]
                f.write(f"    {mid!r}: {smiles!r},\n")
            f.write("}\n\n")
            f.write("# PDB entries containing each missing chem_comp_id\n")
            f.write("missing_chem_comp_pdbs = {\n")
            for mid in sorted(all_missing_to_pdbs.keys()):
                pdbs = sorted(all_missing_to_pdbs[mid]["pdbs"])
                f.write(f"    {mid!r}: {pdbs!r},\n")
            f.write("}\n")

        print(f"\nResults saved to {result_path}")
    else:
        print("\nNo missing chem_comp_ids found! All IDs have matches in preprocessed_CCD.lmdb.")
        result_path = f"{BASE_DIR}/result.py"
        with open(result_path, "w") as f:
            f.write("# No missing chem_comp_ids found\n")
            f.write("missing_chem_comp_ids = {}\n")
        print(f"Empty result saved to {result_path}")


if __name__ == "__main__":
    main()
