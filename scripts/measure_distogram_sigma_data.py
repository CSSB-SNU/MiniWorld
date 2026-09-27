"""Measure the v1.3 pseudo-beta bin distribution on the configured training mix.

Use a CPU Slurm allocation. The same target helper, crop/atom budgets, and sampler
as training are used. Reports both pair-pooled and equal-per-structure moments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.loss.auxiliary import atom_distogram_target


def measure(cfg, count, workers):
    dataset = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db, sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    ))
    loader = dataset.create_ddp_dataloader(
        world_size=1, rank=0, seed=130, drop_last=True, batch_size=1,
        num_workers=workers, prefetch_factor=2 if workers else None,
        num_samples_per_rank=count, persistent_workers=False, shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple, bucket_template_multiple=4,
    )
    loader.sampler.set_epoch(0)
    total = total_sq = pairs = per_mean = per_sq = top = items = 0
    for batch in loader:
        st, sc = batch.structure, batch.scheme
        if st.atom_is_rep is None:
            raise ValueError("Missing representative-atom labels")
        bins, valid = atom_distogram_target(
            st.atom_pos, st.atom_pos_mask, sc.atom_to_token_idx_map,
            cfg.model.shared.n_distogram_bins, st.token_mask.shape[1],
            rep_atom_mask=st.atom_is_rep,
        )
        valid &= st.token_mask[:, :, None] & st.token_mask[:, None, :]
        values = bins[valid.triu(1)].double()
        if not values.numel(): continue
        items += 1
        pairs += values.numel()
        total += values.sum().item()
        total_sq += values.square().sum().item()
        per_mean += values.mean().item()
        per_sq += values.square().mean().item()
        top += (values == cfg.model.shared.n_distogram_bins - 1).sum().item()
        if items % 16 == 0: print(f"items={items} pairs={pairs}", flush=True)
    if not pairs: raise RuntimeError("No valid distogram targets")
    return {
        "crop_tokens": cfg.data.crop.max_tokens, "crop_atoms": cfg.data.crop.max_atoms,
        "num_bins": cfg.model.shared.n_distogram_bins, "seed": 130,
        "items": items, "pairs": pairs, "pair_pooled_mean": total / pairs,
        "pair_pooled_std": math.sqrt(max(0., total_sq / pairs - (total / pairs)**2)),
        "equal_structure_mean": per_mean / items,
        "equal_structure_std": math.sqrt(max(0., per_sq / items - (per_mean / items)**2)),
        "fraction_overflow": top / pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-items", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with initialize_config_dir(str(args.config.resolve().parent), version_base=None):
        cfg = compose(config_name=args.config.stem)
    result = measure(cfg, args.num_items, args.num_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
