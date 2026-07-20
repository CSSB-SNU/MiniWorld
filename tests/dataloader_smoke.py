"""Dataloader smoke test: build the REAL BioMolData dataloader from the local
(bioai-aligned: atom tokenizer + no_pairing) config exactly as the training script
does, pull a few batches, and validate shapes / finiteness / timing.

Run from MiniWorld root (cwd) under the cu128 env.
"""
import sys, time
sys.path.insert(0, "scripts")
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config
from miniworld.data.dataloader.dataloader import BioMolData

import os
CONFIG = "config_distogram_swa_af3_mix_local"
N_BATCHES = int(os.environ.get("DL_SMOKE_N", "6"))
NUM_WORKERS = int(os.environ.get("DL_SMOKE_WORKERS", "4"))


def finite(x):
    return "n/a" if x is None else ("OK" if torch.isfinite(x.float()).all() else "HAS_NAN/INF")


def main():
    with initialize_config_dir(str(Path("configs/miniworld").absolute()), version_base=None):
        raw = compose(config_name=CONFIG)
    cfg = Config.model_validate(raw)
    print(f"config OK | tokenizer={cfg.data.tokenizer.level} pairing={cfg.data.msa.pairing_mode} "
          f"crop(tok={cfg.data.crop.max_tokens},atom={cfg.data.crop.max_atoms}) msa_depth={cfg.data.msa.max_msa_depth}",
          flush=True)

    t0 = time.perf_counter()
    ds = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    ))
    print(f"BioMolData init: {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    dl = ds.create_ddp_dataloader(
        world_size=1, rank=0, seed=cfg.train.seed, drop_last=True,
        batch_size=cfg.train.num_batch, num_workers=NUM_WORKERS, prefetch_factor=2,
        num_samples_per_rank=N_BATCHES, persistent_workers=False, shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
    )
    print(f"dataloader build: {time.perf_counter()-t1:.1f}s | pulling {N_BATCHES} batches...", flush=True)

    t2 = time.perf_counter(); prev = t2; ok = 0
    for i, b in enumerate(dl):
        now = time.perf_counter()
        s = b.structure
        print(f"  batch {i}: name={str(b.name[0])[:32]:32s} tok={b.token_length:4d} atom={b.atom_length:5d} "
              f"msa={b.msa_count}/{b.msa_depth} tmpl={b.template_count}/{b.template_number} "
              f"| atom_pos={finite(s.atom_pos)} tok_mask_sum={int(s.token_mask.sum())} "
              f"| {now-prev:.2f}s", flush=True)
        prev = now; ok += 1
        if i + 1 >= N_BATCHES:
            break

    print(f"\nDATALOADER_SMOKE_DONE ok={ok}/{N_BATCHES} total_iter={time.perf_counter()-t2:.1f}s", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
