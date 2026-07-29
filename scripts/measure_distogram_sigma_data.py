"""Measure EDM ``sigma_data`` for the distogram-diffusion target ACROSS crop sizes.

The diffusion "image" is the per-pair distogram BIN INDEX (0..D-1) of the CB/pseudo-beta
representative-atom distance, bucketised with ``edges = linspace(min_d, max_d, D-1)``
(identical to ``cal_atom_distogram_loss`` with ``rep_atom_mask``). EDM's ``sigma_data`` is
the marginal std of a pixel, and it depends on the crop size (a bigger L is more
far-heavy), so this sweeps several ``crop.max_tokens`` and reports the pooled std for each.

Distances are computed at the TOKEN level: each token's single rep atom (CB) is scattered
to a token slot and ``cdist`` gives the [L, L] rep-rep distances — O(L^2), so crop=2048 is
cheap (the atom^2 path would blow up).

Run (CPU node)::

    .pixi/envs/cu128/bin/python scripts/measure_distogram_sigma_data.py \
        --config configs/miniworld/config_distogram_swa_bioai_small_pf4_d512.yaml \
        --crops 384,512,768,1024,2048 --num-items 800 --num-workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData

MIN_DISTANCE = 2.25
MAX_DISTANCE = 25.75
ATOM_MULT = 12  # max_atoms budget per token so the token crop binds, not the atom crop


def _bins_for_item(
    atom_pos: torch.Tensor,        # [L_atom, 3]
    atom_pos_mask: torch.Tensor,   # [L_atom] bool
    atom_is_rep: torch.Tensor,     # [L_atom] bool
    atom_to_token: torch.Tensor,   # [L_atom] int
    token_num: int,
    edges: torch.Tensor,           # [D-1]
) -> tuple[torch.Tensor, int]:
    """Token-level CB distogram bins over valid i<j pairs, + n_valid residues."""
    rep = atom_pos_mask.bool() & atom_is_rep.bool()  # [L_atom]
    idx = atom_to_token.clamp(0, token_num - 1)
    tok_pos = torch.zeros(token_num, 3, dtype=torch.float32)
    tok_valid = torch.zeros(token_num, dtype=torch.bool)
    tok_pos[idx[rep]] = atom_pos[rep].float()  # one rep atom (CB) per token
    tok_valid[idx[rep]] = True

    dists = torch.cdist(tok_pos[None], tok_pos[None])[0]  # [L, L]
    target = torch.bucketize(dists, edges)  # [L, L] int in [0, D-1]
    pair_mask = tok_valid[:, None] & tok_valid[None, :]
    tri = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    valid = pair_mask & tri
    return target[valid].to(torch.int64), int(tok_valid.sum().item())


def _measure_one_crop(cfg, crop_tokens: int, num_items: int, num_workers: int) -> dict:
    """Build the dataset at ``crop_tokens`` and return pooled bin stats."""
    OmegaConf.set_struct(cfg, value=False)
    cfg.data.crop.max_tokens = crop_tokens
    cfg.data.crop.max_atoms = crop_tokens * ATOM_MULT

    D = int(cfg.model.shared.n_distogram_bins)
    edges = torch.linspace(MIN_DISTANCE, MAX_DISTANCE, D - 1)

    data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )
    dataset = BioMolData(data_config)
    loader = dataset.create_ddp_dataloader(
        world_size=1, rank=0, seed=0, drop_last=True, batch_size=1,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        num_samples_per_rank=num_items,
        persistent_workers=False, shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
        bucket_template_multiple=4,
    )
    loader.sampler.set_epoch(0)

    n_total, s1, s2, top, n_items = 0, 0.0, 0.0, 0, 0
    max_l = 0
    for batch in loader:
        struct, scheme = batch.structure, batch.scheme
        for i in range(struct.atom_pos.shape[0]):
            if struct.atom_is_rep is None:
                print("[measure] atom_is_rep is None; abort")
                return {}
            token_num = int(struct.token_mask[i].shape[-1])
            bins, n_valid = _bins_for_item(
                struct.atom_pos[i], struct.atom_pos_mask[i],
                struct.atom_is_rep[i], scheme.atom_to_token_idx_map[i],
                token_num, edges,
            )
            if bins.numel() == 0:
                continue
            bf = bins.to(torch.float64)
            n_total += bf.numel()
            s1 += bf.sum().item()
            s2 += (bf * bf).sum().item()
            top += int((bins == D - 1).sum().item())
            max_l = max(max_l, n_valid)
            n_items += 1
        if n_items and n_items % 200 == 0:
            m = s1 / n_total
            print(f"[measure] crop={crop_tokens} items={n_items} "
                  f"mean={m:.2f} std={(s2 / n_total - m * m) ** 0.5:.3f}")

    if n_total == 0:
        return {}
    mean = s1 / n_total
    std = max(s2 / n_total - mean * mean, 0.0) ** 0.5
    return {
        "crop": crop_tokens, "items": n_items, "pairs": n_total,
        "mean": mean, "std": std, "frac_top": top / n_total, "max_l": max_l,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--crops", type=str, default="384,512,768,1024,2048")
    ap.add_argument("--num-items", type=int, default=800)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    config = args.config
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name)

    # Measure on the ACTUAL training distribution = the config's source mix (pdb +
    # distillation), since that is what the diffusion model trains on. The dataloader
    # fix (8a4bd96) is required for source_weights to be honoured on a catalog-cache hit.
    OmegaConf.set_struct(cfg, value=False)
    sw = OmegaConf.to_container(cfg.data.train_db.source_weights, resolve=True)
    print(f"[measure] source mix (as configured)={sw}")
    print(f"[measure] crops={args.crops} num_items={args.num_items}")

    crops = [int(c) for c in args.crops.split(",")]
    D = int(cfg.model.shared.n_distogram_bins)
    norm = 2.0 / (D - 1)
    rows = []
    for crop in crops:
        r = _measure_one_crop(cfg, crop, args.num_items, args.num_workers)
        if r:
            rows.append(r)
            print(f"[RESULT] crop={r['crop']:>5} items={r['items']:>4} "
                  f"max_l={r['max_l']:>5} mean={r['mean']:.2f} "
                  f"sigma_raw={r['std']:.3f} sigma_[-1,1]={r['std'] * norm:.4f} "
                  f"frac_top={r['frac_top']:.3f}")

    print("\n" + "=" * 78)
    print(f"[RESULT] sigma_data vs crop size (PDB, D={D} bins, CB target)")
    print(f"[RESULT] {'crop':>6} {'items':>6} {'max_L':>6} {'mean_bin':>9} "
          f"{'sigma_raw':>10} {'sigma_[-1,1]':>12} {'frac_top':>9}")
    for r in rows:
        print(f"[RESULT] {r['crop']:>6} {r['items']:>6} {r['max_l']:>6} "
              f"{r['mean']:>9.2f} {r['std']:>10.3f} {r['std'] * norm:>12.4f} "
              f"{r['frac_top']:>9.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
