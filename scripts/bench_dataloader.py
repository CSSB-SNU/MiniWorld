"""Dataloader speed benchmark — real BioMolData, current BioAI setup.

Mirrors the training entrypoint's dataloader construction (BioMolData +
create_ddp_dataloader with the same bucket multiples / batch size / prefetch),
then times pulling N items per num_workers setting. Reports dataset-init time
(catalog mmap + CCD), cold first-item latency (worker spawn + pipeline fill),
and steady-state items/s.

CPU-only (no GPU): batches are built on CPU; we do NOT move them to device.

Run (cu128 env, CPU node):
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib PYTHONNOUSERSITE=1 \
    python scripts/bench_dataloader.py \
      --config configs/miniworld/config_distogram_swa_af3_mix_bioai_8gpu.yaml \
      --n 100 --workers 4,8,16
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from run_miniworld_distogram_train import Config  # type: ignore[import-not-found]
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.configs.data import TemplateConfig


def _p(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="items to pull per setting")
    ap.add_argument("--workers", type=str, default="4,8,16", help="comma list of num_workers")
    ap.add_argument("--templates", type=int, default=0, help="n_templates (0 = skip template LMDB reads)")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        raw = compose(config_name=cfg_path.stem)
    cfg = Config.model_validate(raw)

    print("=" * 72)
    print(f"config        : {cfg_path.name}")
    print(f"crop          : max_tokens={cfg.data.crop.max_tokens} max_atoms={cfg.data.crop.max_atoms}")
    print(f"msa           : max_depth={cfg.data.msa.max_msa_depth} sample_depth={cfg.data.msa.sample_depth}")
    print(f"batch_size    : {cfg.train.num_batch}  prefetch_factor={cfg.train.prefetch_factor}")
    print(f"buckets       : msa={cfg.train.bucket_msa_multiple} tok={cfg.train.bucket_token_multiple} atom={cfg.train.bucket_atom_multiple}")
    print("=" * 72)

    t0 = time.perf_counter()
    dataset = BioMolData(
        BioMolData.BioMolConfig(
            crop_config=cfg.data.crop,
            msa_config=cfg.data.msa,
            DB_config=cfg.data.train_db,
            sampler_config=cfg.data.sampler,
            tokenizer_config=cfg.data.tokenizer,
            template_config=TemplateConfig(n_templates=args.templates),
        )
    )
    t_init = time.perf_counter() - t0
    print(f"dataset init  : {t_init:.1f}s   (catalog items = {len(dataset)})")
    print("")

    workers = [int(w) for w in args.workers.split(",") if w.strip()]
    print("%-9s %10s %10s %10s %10s %10s %10s" % (
        "workers", "init(s)", "first(s)", "total(s)", "med(ms)", "p95(ms)", "items/s"))

    for nw in workers:
        t0 = time.perf_counter()
        dl = dataset.create_ddp_dataloader(
            rank=0,
            world_size=1,
            shuffle=True,
            seed=0,
            drop_last=True,
            num_workers=nw,
            num_samples_per_rank=args.n + 50,
            batch_size=cfg.train.num_batch,
            prefetch_factor=cfg.train.prefetch_factor if nw > 0 else None,
            persistent_workers=nw > 0,
            bucket_msa_multiple=cfg.train.bucket_msa_multiple,
            bucket_token_multiple=cfg.train.bucket_token_multiple,
            bucket_atom_multiple=cfg.train.bucket_atom_multiple,
        )
        dataset.set_epoch(0)
        dl.sampler.set_epoch(0)  # type: ignore[attr-defined]
        it = iter(dl)

        per = []
        t_loop = time.perf_counter()
        t_prev = t_loop
        first = None
        for i in range(args.n):
            _b = next(it)
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if i == 0:
                first = dt
            else:
                per.append(dt)
        total = time.perf_counter() - t_loop
        dl_init = t_loop - t0

        med = statistics.median(per) * 1e3
        p95 = _p(per, 0.95) * 1e3
        items_s = args.n / total
        print("%-9d %10.1f %10.2f %10.2f %10.1f %10.1f %10.2f" % (
            nw, dl_init, first, total, med, p95, items_s))

        del it, dl
    print("\nDONE")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
