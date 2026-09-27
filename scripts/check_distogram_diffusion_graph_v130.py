"""Full PF16 diffusion CUDA-graph preflight; run in a GPU Slurm allocation.

Uses two real training batches per rank plus a true T=0-template variant. Checks
64 accumulated microbatches before and after Adam, including graph RNG, loss and
all-reduced gradients against ordinary compiled execution. No W&B run or training
checkpoint is created. The two final diagnostic optimizer steps use cached inputs;
their timings do not measure data-loader throughput.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from run_miniworld_distogram_train import Config
from random_recycle_graph_trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/miniworld/distogram_diffusion_medium_v130_bioai.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    with initialize_config_dir(str(args.config.resolve().parent), version_base=None):
        raw = compose(config_name=args.config.stem, overrides=args.override)
    cfg = Config.model_validate(OmegaConf.to_container(raw, resolve=True))
    assert cfg.model.trunk.diffusion is not None
    assert cfg.model.trunk.n_recycle_max == 1
    assert cfg.loss.distogram_interchain_weight == 2.0
    assert cfg.train.compile
    cfg.train.use_wandb = False
    cfg.train.run_dir = str(args.output_dir)
    # Production warmup starts at lr=0. This diagnostic must actually update the
    # zero-initialized decoder before checking upstream trunk/encoder gradients.
    cfg.train.warmup_steps = 0
    cfg.train.max_lr = 1e-4
    # Only two real examples are needed per rank. Reduce CPU prefetch pressure.
    cfg.train.num_workers = 2
    cfg.train.prefetch_factor = 2
    os.environ["MW_GRAPH_REPEAT_DIAGNOSTIC_INPUTS"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    rank = int(os.environ.get("RANK", "0"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"config-rank{rank}.json").write_text(cfg.model_dump_json(indent=2))
    print(f"[v130 graph] rank={rank} L={cfg.data.crop.max_tokens} "
          f"accumulation={cfg.train.grad_accum_steps}; real batches, no recycling", flush=True)
    train(cfg, None, args.output_dir, diagnostic_steps=2, validate=True)
    checks = json.loads((args.output_dir / f"validation-rank{rank}.json").read_text())
    after = next(x for x in checks if x["stage"] == "after_optimizer")
    grads = after["gradient_details"]
    assert any(x["name"] == "diffusion.encoder.weight" and x["reference_norm"] > 0 for x in grads)
    assert any(x["name"].startswith("pairformer_blocks.") and x["reference_norm"] > 0 for x in grads)
    result = {
        "passed": True, "rank": rank, "tokens": cfg.data.crop.max_tokens,
        "atoms": cfg.data.crop.max_atoms, "accumulation": cfg.train.grad_accum_steps,
        "interchain_weight": cfg.loss.distogram_interchain_weight,
        "diagnostic_lr": cfg.train.max_lr,
        "timing_scope": "compiled fwd+EDM loss+bwd per microbatch; excludes H2D/NCCL/optimizer",
        "checks": [{
            "stage": x["stage"],
            "max_gradient_relative_l2": x["max_gradient_relative_l2"],
            "distinct_sigma": len(set(x["actual_sigma"])),
            "compile_ms_median": statistics.median(x["reference_compute_ms"]),
            "graph_ms_median": statistics.median(x["graph_compute_ms"]),
        } for x in checks],
    }
    (args.output_dir / f"summary-rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
