"""
Re-fetch canonical SMILES for all missing chem_comp_ids in result.py.
The original fetch was buggy (truncated to first char). This script
reads the IDs and PDB mappings from result.py, fetches correct SMILES,
and overwrites result.py with correct data.
"""

import json
import re
import time
import urllib.error
import urllib.request

RESULT_PATH = "/public_data/bsoohyuncd/BioMolDB_20260224/result.py"


def load_existing_result():
    """Parse missing IDs and PDB mappings from existing result.py."""
    with open(RESULT_PATH) as f:
        content = f.read()

    # Execute to get the dicts
    ns = {}
    exec(content, ns)
    return ns["missing_chem_comp_ids"], ns["missing_chem_comp_pdbs"]


def fetch_smiles(comp_id):
    """Fetch canonical SMILES from PDB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        desc = data.get("rcsb_chem_comp_descriptor", {})

        # Prefer smilesstereo (canonical with stereochemistry)
        if desc.get("smilesstereo"):
            return desc["smilesstereo"]

        # Fallback to smiles
        if desc.get("smiles"):
            return desc["smiles"]

        # Fallback to pdbx_chem_comp_descriptor
        pdbx = data.get("pdbx_chem_comp_descriptor", [])
        for entry in pdbx:
            if isinstance(entry, dict) and entry.get("type") == "SMILES_CANONICAL":
                return entry.get("descriptor", "")
        for entry in pdbx:
            if isinstance(entry, dict) and "SMILES" in entry.get("type", ""):
                return entry.get("descriptor", "")

        return ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "NOT_FOUND_IN_PDB"
        return f"HTTP_ERROR_{e.code}"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("Loading existing result.py...")
    old_ids, pdb_mapping = load_existing_result()
    comp_ids = sorted(old_ids.keys())
    print(f"  {len(comp_ids)} missing chem_comp_ids to re-fetch.")

    new_ids = {}
    for i, cid in enumerate(comp_ids):
        smiles = fetch_smiles(cid)
        new_ids[cid] = smiles
        status = smiles[:80] if smiles else "N/A"
        print(f"  [{i+1}/{len(comp_ids)}] {cid}: {status}")
        time.sleep(0.05)

    # Write updated result.py
    with open(RESULT_PATH, "w") as f:
        f.write("# Missing chem_comp_ids not found in preprocessed_CCD.lmdb\n")
        f.write("# Keys: chem_comp_id, Values: canonical SMILES from PDB\n\n")
        f.write("missing_chem_comp_ids = {\n")
        for cid in sorted(new_ids.keys()):
            smiles = new_ids[cid]
            f.write(f"    {cid!r}: {smiles!r},\n")
        f.write("}\n\n")
        f.write("# PDB entries containing each missing chem_comp_id\n")
        f.write("missing_chem_comp_pdbs = {\n")
        for cid in sorted(pdb_mapping.keys()):
            pdbs = pdb_mapping[cid]
            if isinstance(pdbs, list):
                pdbs = sorted(pdbs)
            f.write(f"    {cid!r}: {pdbs!r},\n")
        f.write("}\n")

    print(f"\nUpdated {RESULT_PATH} with correct SMILES.")


if __name__ == "__main__":
    main()
