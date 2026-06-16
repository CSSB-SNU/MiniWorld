"""Per-submodule fwd/bwd profiling of a SINGLE Pairformer block.

Isolates one ``PairformerBlock`` (block 0 of the trunk-configured stack) and
times every sub-module separately for forward (CUDA events) and backward
(autograd identity markers), for both ``use_single=True`` and ``False``.

A PairformerBlock is:
    pair = pair + drop_row(tri_multi_outgoing(pair))
    pair = pair + drop_row(tri_multi_incoming(pair))
    pair = pair + drop_row(tri_atten_starting(pair))
    pair = pair + drop_col(tri_atten_ending(pair))
    pair = pair + transition_pair(pair)
    if use_single:                              # the part toggled off
        single = single + pair_to_single(single, pair)
        single = single + transition_single(single)

Reuses the Timers / wrap_module / marker machinery from profile_miniworld_edm.

Usage (one H100):
    pixi run -e cu128 python scripts/profile_pairformer_block.py \
        --config configs/miniworld/config_exp_msa3_24_3_edm.yaml \
        --model large_msa3_24_3 --n-iters 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_miniworld_edm import Timers, wrap_module
from run_miniworld_edm_only_train import Config

from team_gm.modules import Pairformer

torch.set_float32_matmul_precision("medium")

# Sub-modules timed inside a block. pair_to_single / transition_single only
# exist when use_single=True.
PAIR_MODULES = [
    "tri_multi_outgoing",
    "tri_multi_incoming",
    "tri_atten_starting",
    "tri_atten_ending",
    "transition_pair",
]
SINGLE_MODULES = ["pair_to_single", "transition_single"]


def _run(
    pcfg: Pairformer.Config,
    *,
    n_tokens: int,
    n_iters: int,
    device: torch.device,
) -> dict:
    """Build a one-block Pairformer with this config and time its submodules."""
    block_cfg = pcfg.model_copy(update={"n_block": 1, "n_checkpoint_segments": None})
    pf = Pairformer(block_cfg).to(device).to(torch.bfloat16)
    pf.train()
    block = pf.pairformer_blocks[0]

    names = list(PAIR_MODULES)
    if pcfg.use_single:
        names += SINGLE_MODULES

    timers = Timers()
    restores = [wrap_module(getattr(block, n), n, timers) for n in names]

    d_pair = pcfg.d_pair
    d_single = pcfg.d_single
    pair0 = torch.randn(1, n_tokens, n_tokens, d_pair, device=device, dtype=torch.bfloat16)
    single0 = torch.randn(1, n_tokens, d_single, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, n_tokens, dtype=torch.bool, device=device)

    measured: list[dict] = []
    try:
        for i in range(n_iters):
            timers.reset()
            pair = pair0.detach().clone().requires_grad_(True)
            single = single0.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            fs = torch.cuda.Event(enable_timing=True)
            fe = torch.cuda.Event(enable_timing=True)
            bs = torch.cuda.Event(enable_timing=True)
            be = torch.cuda.Event(enable_timing=True)
            fs.record()
            out_pair, out_single = block(pair, single, mask)
            fe.record()
            loss = out_pair.float().pow(2).mean()
            if out_single is not None:
                loss = loss + out_single.float().pow(2).mean()
            bs.record()
            loss.backward()
            be.record()
            torch.cuda.synchronize()
            rec = {
                "_wall_fwd": fs.elapsed_time(fe),
                "_wall_bwd": bs.elapsed_time(be),
                **{f"{n}_fwd": timers.fwd_ms(f"{n}/grad") for n in names},
                **{f"{n}_bwd": timers.bwd_ms(n) for n in names},
            }
            if i > 0:  # iter 0 is warmup (kernel autotune)
                measured.append(rec)
    finally:
        for r in restores:
            r()
    del pf

    # mean over measured iters
    keys = measured[0].keys()
    return {k: sum(m[k] for m in measured) / len(measured) for k in keys}


def _print(label: str, res: dict, names: list[str]) -> None:
    click.echo(f"\n=== {label} ===")
    click.echo(f"  {'submodule':<20}{'fwd (ms)':>12}{'bwd (ms)':>12}{'sum (ms)':>12}")
    sf = sb = 0.0
    for n in names:
        f = res[f"{n}_fwd"]
        b = res[f"{n}_bwd"]
        sf += f
        sb += b
        click.echo(f"  {n:<20}{f:>12.3f}{b:>12.3f}{f + b:>12.3f}")
    wf = res["_wall_fwd"]
    wb = res["_wall_bwd"]
    click.echo(f"  {'-' * 56}")
    click.echo(f"  {'sum(submodules)':<20}{sf:>12.3f}{sb:>12.3f}{sf + sb:>12.3f}")
    click.echo(f"  {'other (resid/drop)':<20}{wf - sf:>12.3f}{wb - sb:>12.3f}")
    click.echo(f"  {'BLOCK wall':<20}{wf:>12.3f}{wb:>12.3f}{wf + wb:>12.3f}")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/miniworld/config_exp_msa3_24_3_edm.yaml"),
    show_default=True,
)
@click.option("--model", "model_override", type=str, default="large_msa3_24_3", show_default=True)
@click.option("--n-tokens", type=int, default=384, show_default=True)
@click.option("--n-iters", type=int, default=5, show_default=True, help="First iter is warmup.")
def main(config: Path, model_override: str, n_tokens: int, n_iters: int) -> None:
    """Profile a single Pairformer block, use_single True vs False."""
    from hydra import compose, initialize_config_dir

    if not torch.cuda.is_available():
        msg = "CUDA is required."
        raise RuntimeError(msg)
    device = torch.device("cuda")

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg_raw = compose(config_name=config.name, overrides=[f"model={model_override}"])
    cfg = Config.model_validate(cfg_raw)
    base = cfg.model.trunk.pairformer

    click.echo(
        f"Pairformer block profile: L={n_tokens} d_pair={base.d_pair} "
        f"d_single={base.d_single} impl={base.implementation} "
        f"self_attn={base.use_self_attention} gpu={torch.cuda.get_device_name(0)}",
    )

    for use_single in (True, False):
        pcfg = base.model_copy(update={"use_single": use_single})
        names = list(PAIR_MODULES) + (SINGLE_MODULES if use_single else [])
        res = _run(pcfg, n_tokens=n_tokens, n_iters=n_iters, device=device)
        _print(f"use_single={use_single}", res, names)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
