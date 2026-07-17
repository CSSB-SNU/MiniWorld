"""Synthetic-dataloader speed benchmark for the distogram-only Mini model.

Reuses the training entrypoint's own machinery (Config, Client, and the
`_build_precompile_batch` synthetic Batch builder used for CUDA-graph warmup) so
the timed path is exactly the real train step: loss_fn(batch) [fwd] + backward +
optimizer.step. No real data / dataloader — a single dense synthetic batch at the
padded bucket shapes is reused every iteration (the graph-capture regime).

Run (single GPU, cu128 env, on an H100/H200):
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib PYTHONNOUSERSITE=1 \
    python scripts/bench_distogram_synthetic.py \
      --config configs/miniworld/config_distogram_swa_af3_mix_bioai_8gpu.yaml \
      --recycle 1,4 --iters 30 --warmup 8
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric

# Reuse the training entrypoint's helpers (this file sits next to it in scripts/).
from run_miniworld_distogram_train import (  # type: ignore[import-not-found]
    Config,
    _build_precompile_batch,
    _find_recycle_model,
)

from miniworld.configs import TemplateConfig
from miniworld.models.distogram_only import Client
from miniworld.utils import get_step_decay_scheduler_with_warmup


def _parse_recycles(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="top-level hydra config file")
    ap.add_argument("--recycle", type=str, default="1,4", help="comma list of n_recycle to time")
    ap.add_argument("--iters", type=int, default=30, help="timed iterations per recycle")
    ap.add_argument("--warmup", type=int, default=8, help="warmup iterations (compile/graph capture)")
    ap.add_argument("--no-compile", action="store_true", help="force-disable cfg.train.compile")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        raw = compose(config_name=cfg_path.stem)
    cfg = Config.model_validate(raw)

    fabric = Fabric(devices=1)
    fabric.launch()
    device = fabric.device

    client = Client(Client.Config(train=cfg.train, model=cfg.model, loss=cfg.loss))

    compile_on = cfg.train.compile and not args.no_compile
    if compile_on:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)

    optimizer = torch.optim.Adam(client.model.parameters(), cfg.train.max_lr, betas=(0.9, 0.95))
    scheduler = get_step_decay_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_steps=cfg.train.warmup_steps,
        decay_steps=cfg.train.decay_steps,
        decay_factor=cfg.train.decay_factor,
    )
    client.setup(
        fabric=fabric,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=1,
        gradient_clip_norm=cfg.train.grad_clip_max_norm,
    )

    raw_model = _find_recycle_model(client.model)
    n_templates = TemplateConfig().n_templates
    msa_depth = cfg.data.msa.max_msa_depth
    n_tokens = cfg.data.crop.max_tokens
    n_atoms = cfg.data.crop.max_atoms
    num_res_class = cfg.model.shared.num_res_class

    print("=" * 72)
    print(f"device        : {torch.cuda.get_device_name(0)}")
    print(f"torch         : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"compile       : {compile_on}")
    print(f"shapes        : msa={msa_depth} tokens={n_tokens} atoms={n_atoms} templ={n_templates}")
    print(f"num_batch     : {cfg.train.num_batch}  (per-GPU)")
    print("=" * 72)

    batch = _build_precompile_batch(
        device=device,
        msa_depth=msa_depth,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_templates=n_templates,
        num_res_class=num_res_class,
    )

    def one_step() -> None:
        client.training_step(batch)
        client.optimizer.step()
        client.optimizer.zero_grad(set_to_none=True)

    results = {}
    client.model.train()
    for nr in _parse_recycles(args.recycle):
        raw_model._forced_n_recycle = nr  # noqa: SLF001
        # warmup (compile + cuda-graph capture happen here)
        client.optimizer.zero_grad(set_to_none=True)
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        times_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1e3)

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        med = statistics.median(times_ms)
        mean = statistics.mean(times_ms)
        p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
        steps_s = 1000.0 / med
        tok_s = steps_s * n_tokens * cfg.train.num_batch
        results[nr] = (med, mean, p95, steps_s, tok_s, peak_gb)
        raw_model._forced_n_recycle = None  # noqa: SLF001

    print("\n%-8s %10s %10s %10s %10s %12s %10s" % (
        "recycle", "med(ms)", "mean(ms)", "p95(ms)", "steps/s", "tokens/s", "peakGB"))
    for nr, (med, mean, p95, steps_s, tok_s, peak_gb) in results.items():
        print("%-8d %10.1f %10.1f %10.1f %10.3f %12.0f %10.2f" % (
            nr, med, mean, p95, steps_s, tok_s, peak_gb))
    print("\nDONE")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
