"""Audit which chem_comp_ids fall back to UNK in the fingerprint vocab.

For each residue in sampled structures, checks whether its chem_comp_id
exists in the fingerprint vocab. If it doesn't (i.e. maps to UNK),
prints the original chem_comp_id and the entity type of that chain.

Usage:
    pixi run python scripts/audit_unk_chem_comp.py \
        --n-samples 500
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from miniworld.configs.data_explicit import BioMolDBConfig, CropConfig, TokenizerConfig
from miniworld.data.dataloader.dataloader_explicit import BioMolData
from miniworld.data.io import load_cifmol
from miniworld.data.pipeline import Tokenizer
from miniworld.data.pipeline.utils import remove_terminal_oxygen


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UNK chem_comp_ids in fp vocab")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of structures to sample")
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="/public_data/bsoohyuncd/fp_emb_vocab.json",
    )
    parser.add_argument(
        "--cif-db-path",
        type=str,
        default="/public_data/bsoohyuncd/BioMolDB_20260224/cif_attached_train.lmdb",
    )
    parser.add_argument(
        "--edge-path",
        type=str,
        default="/public_data/bsoohyuncd/BioMolDB_20260224/metadata/train_edge_node.tsv",
    )
    args = parser.parse_args()

    # Load vocab
    with open(args.vocab_path) as f:
        vocab: dict[str, int] = json.load(f)
    unk_idx = vocab.get("UNK")
    print(f"Vocab size: {len(vocab)}, UNK index: {unk_idx}")

    # Load edge list (same as BioMolData._load_edge_to_cif_ids)
    edge_id_to_bias: dict[str, list[str]] = {}
    with open(args.edge_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key1, key2, value = line.split("\t")
            edge_id = key1 if key2 == "None" else f"{key1}_{key2}"
            edge_id_to_bias[edge_id] = value.split(",")
    edge_id_list = list(edge_id_to_bias.keys())
    print(f"Total edges: {len(edge_id_list)}")

    # Counters
    unk_by_entity: Counter[str] = Counter()           # entity_tag -> count
    unk_chemcomp: Counter[str] = Counter()             # chem_comp_id -> count
    unk_pair: Counter[tuple[str, str]] = Counter()     # (chem_comp_id, entity_tag) -> count
    total_residues = 0
    total_unk = 0
    failed = 0

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    for i in range(args.n_samples):
        try:
            idx = rng.randrange(len(edge_id_list))
            edge_id = edge_id_list[idx]
            bias_str = rng.choice(edge_id_to_bias[edge_id])

            pdb_id, assembly_id, model_id, alt_id = bias_str.split("_")[:4]

            cifmol = load_cifmol(
                db_path=Path(args.cif_db_path),
                pdb_id=pdb_id.lower(),
                assembly_id=assembly_id,
                model_id=model_id,
                alt_id=alt_id,
            )
            cifmol = remove_terminal_oxygen(cifmol)

            chem_comp_ids = np.asarray(cifmol.residues.chem_comp_id.value, dtype=object)
            res_to_chain = np.asarray(cifmol.index_table.res_to_chain, dtype=np.int64)

            # Entity tag per chain from seq_id (first character: P, R, D, L, etc.)
            chain_seq_ids = cifmol.chains.seq_id.value
            chain_entity_tags = np.array(
                [str(s)[0] for s in chain_seq_ids], dtype=object
            )

            n_res = len(cifmol.residues)
            total_residues += n_res

            for r in range(n_res):
                chem = str(chem_comp_ids[r])
                if chem not in vocab:
                    entity_tag = str(chain_entity_tags[res_to_chain[r]])
                    total_unk += 1
                    unk_chemcomp[chem] += 1
                    unk_by_entity[entity_tag] += 1
                    unk_pair[(chem, entity_tag)] += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{args.n_samples} samples...")

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Sample {i} failed: {e}")

    # Report
    print("\n" + "=" * 70)
    print("UNK chem_comp_id AUDIT REPORT")
    print("=" * 70)
    print(f"Samples checked:  {args.n_samples}")
    print(f"Samples failed:   {failed}")
    print(f"Total residues:   {total_residues}")
    print(f"Total UNK:        {total_unk} ({total_unk / max(total_residues, 1) * 100:.4f}%)")

    print(f"\n--- UNK count by entity tag ---")
    for tag, count in unk_by_entity.most_common():
        print(f"  {tag}: {count}")

    print(f"\n--- Top 50 UNK chem_comp_ids ---")
    for chem, count in unk_chemcomp.most_common(50):
        print(f"  {chem}: {count}")

    print(f"\n--- Top 50 (chem_comp_id, entity_tag) pairs mapped to UNK ---")
    for (chem, tag), count in unk_pair.most_common(50):
        print(f"  {chem:10s}  entity={tag}  count={count}")

    # Save full results to JSON
    out_path = Path("unk_audit_results.json")
    results = {
        "summary": {
            "n_samples": args.n_samples,
            "failed": failed,
            "total_residues": total_residues,
            "total_unk": total_unk,
            "unk_ratio": total_unk / max(total_residues, 1),
        },
        "unk_by_entity": dict(unk_by_entity.most_common()),
        "unk_chemcomp": dict(unk_chemcomp.most_common()),
        "unk_pairs": {f"{chem}|{tag}": cnt for (chem, tag), cnt in unk_pair.most_common()},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
