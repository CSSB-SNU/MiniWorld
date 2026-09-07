"""Split components.cif.gz into per-component records and pickle them."""

import gzip
import pickle
import sys

from mmcif_min import parse_category

ATOM_COLS = [
    "atom_id",
    "type_symbol",
    "charge",
    "pdbx_aromatic_flag",
    "pdbx_stereo_config",
    "model_Cartn_x",
    "model_Cartn_y",
    "model_Cartn_z",
    "pdbx_model_Cartn_x_ideal",
    "pdbx_model_Cartn_y_ideal",
    "pdbx_model_Cartn_z_ideal",
    "pdbx_leaving_atom_flag",
]
BOND_COLS = ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag", "pdbx_stereo_config"]
COMP_COLS = [
    "id",
    "name",
    "formula",
    "type",
    "pdbx_release_status",
    "pdbx_formal_charge",
    "pdbx_modified_date",
    "pdbx_initial_date",
]


def take(tags, rows, cols):
    if tags is None:
        return None
    idx = {c: tags.index(c) for c in cols if c in tags}
    return {c: [r[i] for r in rows] for c, i in idx.items()}


def parse_block(block_lines):
    comp_t, comp_r = parse_category(block_lines, "_chem_comp")
    if comp_t is None:
        return None, None
    comp = take(comp_t, comp_r, COMP_COLS)
    cid = comp["id"][0]
    atom_t, atom_r = parse_category(block_lines, "_chem_comp_atom")
    bond_t, bond_r = parse_category(block_lines, "_chem_comp_bond")
    return cid, {
        "comp": {k: v[0] for k, v in comp.items()},
        "atoms": take(atom_t, atom_r, ATOM_COLS),
        "bonds": take(bond_t, bond_r, BOND_COLS),
    }


def main():
    src, out = sys.argv[1], sys.argv[2]
    store = {}
    block = []
    n_err = 0
    errors = []
    with gzip.open(src, "rt", errors="replace") as fh:
        for line in fh:
            # components.cif pads lines with trailing spaces for column
            # alignment; the per-component ligand files do not. Strip it so
            # semicolon text fields come out as they do from the split files.
            line = line.rstrip()
            if line.startswith("data_"):
                if block:
                    try:
                        cid, rec = parse_block(block)
                        if cid is not None:
                            store[cid] = rec
                    except Exception as exc:  # noqa: BLE001
                        n_err += 1
                        errors.append((block[0], f"{type(exc).__name__}: {exc}"))
                block = [line]
            else:
                block.append(line)
    if block:
        try:
            cid, rec = parse_block(block)
            if cid is not None:
                store[cid] = rec
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            errors.append((block[0], f"{type(exc).__name__}: {exc}"))

    print(f"parsed {len(store)} components, {n_err} block errors")
    for e in errors[:10]:
        print("  ", e)
    n_no_atoms = sum(1 for r in store.values() if not r["atoms"])
    n_no_bonds = sum(1 for r in store.values() if not r["bonds"])
    print(f"no atom table: {n_no_atoms}, no bond table: {n_no_bonds}")
    with open(out, "wb") as fh:
        pickle.dump(store, fh, protocol=4)


if __name__ == "__main__":
    main()
