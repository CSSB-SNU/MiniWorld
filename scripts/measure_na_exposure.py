"""Measure how much nucleic acid the model ACTUALLY sees, after sampling AND cropping.

The config weights say what fraction of ITEMS should contain nucleic acid. What the
model trains on is TOKENS in a 384-token crop, and a protein-RNA item cropped around
its protein chain can carry almost no RNA. This draws from the real WeightedSampler,
runs the real preprocessing, and counts tokens by entity type.

CPU only, but heavy (each item reads a CIF and its MSA): run it as a batch job.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The top-level Config model lives in the trainer, not in miniworld.configs -- reuse it
# so this measures exactly the config the training run validates.
from run_miniworld_diffusion_train import Config  # noqa: E402

from miniworld.data.dataloader.dataloader import BioMolData  # noqa: E402

# EntityMapping: 0 ANTIBODY, 1 PROTEIN, 2 DPROTEIN, 3 RNA, 4 DNA, 5 NA, 6 LIGAND, 7 BRANCHED
NAMES = {0: "antibody", 1: "protein", 2: "dprotein", 3: "RNA", 4: "DNA",
         5: "NA", 6: "ligand", 7: "branched"}
NUCLEIC = {3, 4, 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2a_diffusion.yaml")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=cfg_path.name))

    data = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    ))
    loader = data.create_ddp_dataloader(
        world_size=1, rank=0, seed=0, drop_last=False, batch_size=1,
        num_workers=args.workers, num_samples_per_rank=args.n, shuffle=True,
        bucket_msa_multiple=None, bucket_token_multiple=None,
        bucket_atom_multiple=None, bucket_template_multiple=4,
    )
    data.set_epoch(0)
    print(f"catalog items: {len(data.weights)}   drawing {args.n}")

    tok_by_type = collections.Counter()
    items_with = collections.Counter()
    n_items = n_tok = 0
    for batch in loader:
        et = batch.chain.entity_type[0]                 # [L_chain]
        a2c = batch.scheme.token_asym_id[0]             # [L_token] -> chain idx
        mask = batch.structure.token_mask[0].bool()
        types = et[a2c][mask]
        n_items += 1
        n_tok += int(mask.sum())
        present = set()
        for t in types.tolist():
            tok_by_type[t] += 1
            present.add(t)
        for t in present:
            items_with[t] += 1
        if n_items >= args.n:
            break

    print(f"\nitems drawn: {n_items}   tokens: {n_tok}")
    print(f"\n{'entity type':<12}{'tokens':>10}{'% of tokens':>13}{'items containing':>18}{'% of items':>12}")
    print("-" * 66)
    for t, c in sorted(tok_by_type.items(), key=lambda kv: -kv[1]):
        print(f"{NAMES.get(t, t):<12}{c:>10}{100 * c / n_tok:>12.2f}%"
              f"{items_with[t]:>18}{100 * items_with[t] / n_items:>11.1f}%")
    na_tok = sum(c for t, c in tok_by_type.items() if t in NUCLEIC)
    print("-" * 66)
    print(f"{'NUCLEIC':<12}{na_tok:>10}{100 * na_tok / n_tok:>12.2f}%")
    print(f"{'  RNA only':<12}{tok_by_type[3]:>10}{100 * tok_by_type[3] / n_tok:>12.2f}%")
    print(f"{'  DNA only':<12}{tok_by_type[4]:>10}{100 * tok_by_type[4] / n_tok:>12.2f}%")


if __name__ == "__main__":
    main()
