"""Module-level forward / backward profiling for the distogram-only model.

Replays ``Model.forward`` step-by-step with CUDA-event timing around every
major sub-module so we can see where wall time goes. Defaults match the
v0.1 8GPU training shape (large model, max bucket, n_recycle=4), but
runs single-process on one GPU.

Usage:
    pixi run -e cu128 python scripts/profile_distogram_only.py \
        --config configs/miniworld/config_distogram.yaml \
        --model large_distogram --n-iters 3 --per-block
"""

from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir

# Reuse training-script utilities for config schema + synthetic batch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_distogram_train import Config, _build_precompile_batch

from miniworld.configs.data import TemplateConfig
from miniworld.models.distogram_only import Model
from miniworld.modules.msa_util import init_msa, init_token_single_msa

torch.set_float32_matmul_precision("medium")


@contextlib.contextmanager
def cuda_timer(name: str, events: dict[str, list[tuple]]):
    """Record CUDA events around a region; elapsed time computed after sync."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    events[name].append((start, end))


def _instrumented_forward(
    model: Model,
    batch,  # noqa: ANN001
    *,
    n_recycle: int,
    per_block: bool,
    events: dict[str, list[tuple]],
) -> torch.Tensor:
    """Replay Model.forward with CUDA-event timing around each sub-module."""
    cfg = model.config
    msa = batch.msa
    reference = batch.reference
    scheme = batch.scheme
    sequence = batch.sequence
    structure = batch.structure

    with cuda_timer("init_token_single_msa", events):
        token_single_msa = init_token_single_msa(
            msa,
            sequence,
            num_res_class=cfg.shared.num_res_class,
        )

    with cuda_timer("input_feature_embedder", events):
        ts_input, _, tp_init = model.input_feature_embedder(
            token_single_msa,
            reference,
            scheme,
            structure,
        )

    token_mask = structure.token_mask
    token_pair = torch.zeros_like(tp_init).to(torch.bfloat16)
    tp_init_bf = tp_init.to(torch.bfloat16)
    ts_input_bf = ts_input.to(torch.bfloat16)

    with cuda_timer("init_msa", events):
        msa_feat, msa_mask = init_msa(
            msa,
            num_res_class=cfg.shared.num_res_class,
            dtype=torch.bfloat16,
        )

    for i_cycle in range(n_recycle):
        is_last = i_cycle == n_recycle - 1
        grad_ctx = contextlib.nullcontext() if is_last else torch.no_grad()
        with grad_ctx:
            with cuda_timer(f"add_pair_recycle[c{i_cycle}]", events):
                token_pair = tp_init_bf + model.add_pair_recycle(token_pair)
            with cuda_timer(f"msa_module[c{i_cycle}]", events):
                token_pair = token_pair + model.msa_module(
                    msa_feat,
                    msa_mask,
                    token_pair,
                    ts_input_bf,
                    token_mask,
                )
            if per_block:
                # Iterate the inner ModuleList directly so we time each block.
                # We DO need to honor n_checkpoint_segments for fairness when
                # comparing to training, but profiling at block granularity
                # requires walking blocks one-by-one, so checkpointing is off
                # for this path.
                for j, block in enumerate(
                    model.pairformer_blocks.pairformer_blocks,
                ):
                    with cuda_timer(f"pairformer[c{i_cycle}].block{j:02d}", events):
                        token_pair, _ = block(token_pair, None, token_mask)
            else:
                with cuda_timer(f"pairformer[c{i_cycle}]", events):
                    token_pair, _ = model.pairformer_blocks(
                        token_pair,
                        None,
                        token_mask,
                    )

    with cuda_timer("distogram_head", events):
        return model.distogram_head(token_pair)


def _summarize(events: dict[str, list[tuple]]) -> dict[str, float]:
    """Convert recorded events to elapsed ms (sums across multiple records)."""
    return {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        for name, pairs in events.items()
    }


def _print_iter(label: str, totals: dict[str, float]) -> None:
    click.echo(f"\n=== {label} ===")
    full_fwd = totals.pop("__full_forward", 0.0)
    backward = totals.pop("__backward", 0.0)
    grand = full_fwd + backward
    rows = sorted(totals.items(), key=lambda kv: -kv[1])
    for name, ms in rows:
        pct = (ms / grand * 100) if grand else 0.0
        click.echo(f"  {ms:9.2f} ms ({pct:5.1f}%)  {name}")
    click.echo(
        f"  --- forward sum (instrumented): "
        f"{sum(ms for _, ms in rows):9.2f} ms",
    )
    click.echo(f"  --- full forward (wall):       {full_fwd:9.2f} ms")
    click.echo(f"  --- backward (wall):           {backward:9.2f} ms")
    click.echo(f"  --- TOTAL:                     {grand:9.2f} ms")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/miniworld/config_distogram.yaml"),
    show_default=True,
)
@click.option(
    "--model",
    "model_override",
    type=str,
    default="large_distogram",
    show_default=True,
    help="Hydra override for the model config (e.g. small_distogram).",
)
@click.option("--n-recycle", type=int, default=4, show_default=True)
@click.option(
    "--n-iters",
    type=int,
    default=3,
    show_default=True,
    help="Total iterations; first is warmup, rest are measured.",
)
@click.option(
    "--per-block",
    is_flag=True,
    default=False,
    help="Time each pairformer block individually.",
)
@click.option(
    "--forward-only",
    is_flag=True,
    default=False,
    help="Skip backward pass.",
)
@click.option(
    "--trace",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to dump a chrome trace from torch.profiler.",
)
def main(
    config: Path,
    model_override: str,
    n_recycle: int,
    n_iters: int,
    per_block: bool,
    forward_only: bool,
    trace: Path | None,
) -> None:
    """Profile each sub-module of the distogram-only Model."""
    if not torch.cuda.is_available():
        msg = "CUDA is required for this profile."
        raise RuntimeError(msg)
    device = torch.device("cuda")

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg_raw = compose(
            config_name=config.name,
            overrides=[f"model={model_override}", "train=distogram_v0.1"],
        )
    cfg = Config.model_validate(cfg_raw)

    click.echo(
        f"Profile config: model={model_override} "
        f"n_recycle={n_recycle} per_block={per_block} forward_only={forward_only}",
    )
    click.echo(
        f"Shape: msa={cfg.data.msa.max_msa_depth} "
        f"tokens={cfg.data.crop.max_tokens} atoms={cfg.data.crop.max_atoms}",
    )

    model = Model(cfg.model).to(device)
    model.train()
    model._forced_n_recycle = n_recycle  # noqa: SLF001

    n_templates = TemplateConfig().n_templates
    batch = _build_precompile_batch(
        device=device,
        msa_depth=cfg.data.msa.max_msa_depth,
        n_tokens=cfg.data.crop.max_tokens,
        n_atoms=cfg.data.crop.max_atoms,
        n_templates=n_templates,
        num_res_class=cfg.model.shared.num_res_class,
    )

    measured_totals: list[dict[str, float]] = []

    def one_iter(label: str) -> None:
        events: dict[str, list[tuple]] = defaultdict(list)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with cuda_timer("__full_forward", events):
            logits = _instrumented_forward(
                model,
                batch,
                n_recycle=n_recycle,
                per_block=per_block,
                events=events,
            )
        if not forward_only:
            with cuda_timer("__backward", events):
                logits.sum().backward()
        torch.cuda.synchronize()
        totals = _summarize(events)
        _print_iter(label, totals)
        if not label.startswith("warmup"):
            measured_totals.append(totals)

    one_iter("warmup")
    for i in range(1, n_iters):
        one_iter(f"iter {i}")

    if measured_totals:
        # Aggregate across measured iters: mean per key.
        agg: dict[str, list[float]] = defaultdict(list)
        for it in measured_totals:
            for k, v in it.items():
                agg[k].append(v)
        click.echo("\n=== mean over measured iters ===")
        means = {k: sum(v) / len(v) for k, v in agg.items()}
        _print_iter("mean", means)

    if trace is not None:
        click.echo(f"\nRecording chrome trace to {trace} ...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_modules=True,
        ) as prof:
            logits = _instrumented_forward(
                model,
                batch,
                n_recycle=n_recycle,
                per_block=False,
                events=defaultdict(list),
            )
            if not forward_only:
                logits.sum().backward()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        click.echo(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=30),
        )


if __name__ == "__main__":
    main()
