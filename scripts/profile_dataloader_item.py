"""Per-item preprocessing profiler for the BioMolData dataloader.

Runs the real dataset __getitem__ (num_workers=0, in-process) over N sampler-drawn
items under cProfile, plus a manual stage breakdown that separates LMDB I/O
(load_raw_data) from CPU work (tokenize / crop / MSA build). Tells us whether the
~1.5-2 s/item throughput is shared-FS I/O bound or CPU bound.

Run (cu128 env, CPU node):
  python scripts/profile_dataloader_item.py \
    --config configs/miniworld/config_distogram_swa_af3_mix_bioai_8gpu.yaml --n 40
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict
from pathlib import Path

import torch

from run_miniworld_distogram_train import Config  # type: ignore[import-not-found]
from miniworld.data.dataloader.dataloader import BioMolData
import miniworld.data.io.load as _loadmod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        raw = compose(config_name=cfg_path.stem)
    cfg = Config.model_validate(raw)

    dataset = BioMolData(
        BioMolData.BioMolConfig(
            crop_config=cfg.data.crop,
            msa_config=cfg.data.msa,
            DB_config=cfg.data.train_db,
            sampler_config=cfg.data.sampler,
            tokenizer_config=cfg.data.tokenizer,
        )
    )
    print(f"dataset ready: {len(dataset)} items")

    # ---- manual I/O timer: wrap load_raw_data to accumulate time + bytes per db ----
    io_time: dict[str, float] = defaultdict(float)
    io_calls: dict[str, int] = defaultdict(int)
    io_bytes: dict[str, int] = defaultdict(int)
    _orig = _loadmod.load_raw_data

    def _timed_load_raw_data(key, env_path):  # noqa: ANN001, ANN202
        db = Path(env_path).name
        t0 = time.perf_counter()
        out = _orig(key, env_path)
        io_time[db] += time.perf_counter() - t0
        io_calls[db] += 1
        if out is not None:
            io_bytes[db] += len(out)
        return out

    _loadmod.load_raw_data = _timed_load_raw_data
    # loading.py did `from ...io import load_raw_data` — patch that binding too
    import miniworld.data.dataloader.loading as _lm
    if hasattr(_lm, "load_raw_data"):
        _lm.load_raw_data = _timed_load_raw_data

    dl = dataset.create_ddp_dataloader(
        rank=0, world_size=1, shuffle=True, seed=0, drop_last=True,
        num_workers=0, num_samples_per_rank=args.n + 10, batch_size=cfg.train.num_batch,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
    )
    dataset.set_epoch(0)
    dl.sampler.set_epoch(0)  # type: ignore[attr-defined]
    it = iter(dl)

    # warm one item (first LMDB open / caches), not counted
    next(it)
    io_time.clear(); io_calls.clear(); io_bytes.clear()

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(args.n):
        next(it)
    pr.disable()
    wall = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print(f"{args.n} items  wall={wall:.1f}s  ->  {args.n / wall:.2f} items/s  ({wall / args.n * 1e3:.0f} ms/item)")
    print("=" * 72)
    io_total = sum(io_time.values())
    print(f"\nLMDB I/O total: {io_total:.1f}s  ({100 * io_total / wall:.0f}% of wall)  | CPU/other: {wall - io_total:.1f}s")
    print(f"{'db':40s} {'time(s)':>9} {'calls':>7} {'MB':>8} {'ms/call':>8}")
    for db in sorted(io_time, key=lambda d: -io_time[d]):
        t = io_time[db]; c = io_calls[db]; mb = io_bytes[db] / 1e6
        print(f"{db:40s} {t:9.1f} {c:7d} {mb:8.1f} {1e3 * t / max(c, 1):8.1f}")

    print("\n--- cProfile top 25 by cumulative time ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())

    print("--- cProfile top 20 by total (self) time ---")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(20)
    print(s2.getvalue())
    print("PROFILE DONE")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
