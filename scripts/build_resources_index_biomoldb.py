"""Write the ResourceLocator index (``resources.lmdb``) for a BioMol release tree.

A release laid out like ``BioMolDB_20261014`` keeps every source in one cif / template / MSA LMDB,
so the index is the single ``__meta__`` JSON (source -> base-relative paths) and no per-key rows.

    python scripts/build_resources_index_biomoldb.py --root ~/data/BioMolDB_20261014
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb

# source name (train_item.tsv ``source`` column) -> paths relative to the release root.
SOURCES: dict[str, dict] = {
    "pdb": {
        "cif": "lmdb/pdb/cif_attached/cif_pdb_attached_train_20210930_res9_chain300.lmdb",
        "template": "lmdb/chain/template/template.lmdb",
        "msa": ["lmdb/pdb/msa/a3m.lmdb"],
        "msa_rna": "lmdb/pdb/msa/msa_rna.lmdb",
    },
    "protein_monomer": {
        "cif": "lmdb/distillation_long/cif_attached/cif_long_attached.lmdb",
        "template": "lmdb/distillation_long/template/template_topn_long.lmdb",
        "msa": ["lmdb/distillation_long/msa/msa_long_d2k.lmdb"],
    },
    "short_protein_monomer": {
        "cif": "lmdb/distillation_short/cif_attached/cif_short_attached.lmdb",
        "template": "lmdb/distillation_short/template/template_topn_short.lmdb",
        "msa": ["lmdb/distillation_short/msa/msa_short_d2k.lmdb"],
    },
    "disordered_pdb": {
        "cif": "lmdb/distillation_disordered/cif_attached/cif_disordered_attached.lmdb",
        "template": "lmdb/distillation_disordered/template/template_disordered.lmdb",
        "msa": ["lmdb/distillation_disordered/msa/msa_disordered_d2k.lmdb"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="release root (holds lmdb/)")
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/metadata/resources.lmdb")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = args.out or root / "metadata" / "resources.lmdb"

    sources: dict[str, dict] = {}
    for name, info in SOURCES.items():
        paths = [info["cif"], info["template"], *info["msa"], *([info["msa_rna"]] if "msa_rna" in info else [])]
        missing = [p for p in paths if not (root / p).exists()]
        if missing:
            msg = f"{name}: missing under {root}: {missing}"
            raise FileNotFoundError(msg)
        sources[name] = {**info, "msa_rna": info.get("msa_rna"), "sharded_msa": False}

    meta = {"version": 1, "key_sep": "|", "base": str(root) + "/", "sources": sources}
    out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"__meta__", json.dumps(meta).encode())
    env.close()
    print(f"wrote {out} ({', '.join(sources)})")


if __name__ == "__main__":
    main()
