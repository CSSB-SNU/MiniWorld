"""Module-level forward / backward profiling for the MiniWorld EDM model.

Runs the *real* ``Model.forward`` (trunk recycling + diffusion module) on a
dense synthetic batch shaped like the 3/24/3 EDM training config, and times
the heavy sub-modules separately for forward and backward. The breakdown the
profile reports matches the buckets we care about for compute budgeting:

    1. trunk module  (no grad)   -- recycle cycles 0..n-2 (run under no_grad)
    2. trunk module  (with grad) -- the last recycle cycle (in the autograd graph)
    3. atom dit                  -- atom_transformer in the AtomAttention enc + dec
    4. token dit                 -- the token DiffusionTransformer
    5. etc                       -- everything else (embedders, diffusion
                                    conditioning, atom-pair build, scatter,
                                    distogram head, recycle linears, ...)

How the timing works
--------------------
* Forward: each heavy sub-module is wrapped so a pair of CUDA events brackets
  its forward call. Whether the call is grad-enabled (``torch.is_grad_enabled``)
  decides the no_grad vs with-grad bucket for the trunk, so the same wrapped
  ``msa_module`` / ``pairformer_blocks`` objects serve every recycle cycle.
* Backward: the same wrapper inserts identity autograd "markers" on the
  module's input and output tensors. In the backward pass the output marker
  fires when grad first reaches the module (bwd start) and the input marker
  fires once grad has propagated to its inputs (bwd end); the elapsed time
  between them is that module's backward cost. no_grad cycles contribute no
  backward markers, so trunk-no-grad backward is 0 by construction.
* "etc" forward/backward is the residual: full wall time minus the sum of the
  instrumented modules.

Usage (one H100):
    pixi run -e cu128 python scripts/profile_miniworld_edm.py \
        --config configs/miniworld/config_exp_msa3_24_3_edm.yaml \
        --model large_msa3_24_3 \
        --n-recycle 4 --n-recycle 20 \
        --num-augment 48 --n-iters 3
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import torch

# Reuse training-script utilities for config schema + synthetic batch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_edm_only_train import Config, _build_precompile_batch

from miniworld.configs.data import TemplateConfig
from miniworld.diffusion import EuclideanDiffuser
from miniworld.diffusion.edm.scheduler import EDMScheduler
from miniworld.models.miniworld_edm import Model

torch.set_float32_matmul_precision("medium")


# --------------------------------------------------------------------------
# Autograd markers: identity in forward, record a CUDA event in backward.
# --------------------------------------------------------------------------
class _Marker(torch.autograd.Function):
    """Identity that records a timing event the moment its backward fires."""

    @staticmethod
    def forward(ctx, x, store):  # noqa: ANN001
        ctx.store = store
        return x

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        ctx.store.append(ev)
        return grad, None


def _mark_tensors(obj, store):  # noqa: ANN001
    """Apply a marker to every float tensor in a (possibly nested) structure."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return _Marker.apply(obj, store)
        return obj
    if isinstance(obj, tuple):
        return tuple(_mark_tensors(o, store) for o in obj)
    if isinstance(obj, list):
        return [_mark_tensors(o, store) for o in obj]
    return obj


class Timers:
    """Holds forward CUDA-event pairs and backward marker events per module."""

    def __init__(self) -> None:
        # name -> list[(start_event, end_event)] for forward
        self.fwd: dict[str, list[tuple]] = defaultdict(list)
        # name -> list[event] for backward input-edge markers (bwd end)
        self.bwd_in: dict[str, list] = defaultdict(list)
        # name -> list[event] for backward output markers (bwd start)
        self.bwd_out: dict[str, list] = defaultdict(list)

    def fwd_ms(self, name: str) -> float:
        return sum(s.elapsed_time(e) for s, e in self.fwd.get(name, []))

    def bwd_ms(self, name: str) -> float:
        outs = self.bwd_out.get(name, [])
        ins = self.bwd_in.get(name, [])
        if not outs or not ins:
            return 0.0
        # bwd start = first grad arriving at an output; end = last grad
        # leaving through an input. elapsed_time(start, end).
        return outs[0].elapsed_time(ins[-1])

    def reset(self) -> None:
        self.fwd.clear()
        self.bwd_in.clear()
        self.bwd_out.clear()


def wrap_module(module: torch.nn.Module, name: str, timers: Timers):  # noqa: ANN201
    """Patch ``module.forward`` to record fwd CUDA events + bwd markers.

    The grad-enabled state at call time splits the trunk into ``<name>/grad``
    and ``<name>/nograd`` forward buckets. Returns a restore() callable.
    """
    orig_forward = module.forward

    def patched(*args, **kwargs):  # noqa: ANN002, ANN003
        grad_on = torch.is_grad_enabled()
        bucket = f"{name}/grad" if grad_on else f"{name}/nograd"

        # Input-edge markers (only meaningful when grad is on).
        if grad_on:
            args = tuple(_mark_tensors(a, timers.bwd_in[name]) for a in args)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig_forward(*args, **kwargs)
        end.record()
        timers.fwd[bucket].append((start, end))

        if grad_on:
            out = _mark_tensors(out, timers.bwd_out[name])
        return out

    module.forward = patched  # type: ignore[method-assign]

    def restore() -> None:
        module.forward = orig_forward  # type: ignore[method-assign]

    return restore


# --------------------------------------------------------------------------
# The five reporting buckets, derived from the instrumented module names.
# --------------------------------------------------------------------------
def summarize(timers: Timers, total_fwd_ms: float, total_bwd_ms: float) -> dict:
    """Roll the per-module events up into the five reporting buckets."""
    fwd = timers.fwd_ms
    bwd = timers.bwd_ms

    trunk_wo_fwd = fwd("msa/nograd") + fwd("pairformer/nograd")
    trunk_wg_fwd = fwd("msa/grad") + fwd("pairformer/grad")
    trunk_wg_bwd = bwd("msa") + bwd("pairformer")

    atom_fwd = fwd("atom_dit_enc/grad") + fwd("atom_dit_dec/grad")
    atom_bwd = bwd("atom_dit_enc") + bwd("atom_dit_dec")

    token_fwd = fwd("token_dit/grad")
    token_bwd = bwd("token_dit")

    accounted_fwd = trunk_wo_fwd + trunk_wg_fwd + atom_fwd + token_fwd
    accounted_bwd = trunk_wg_bwd + atom_bwd + token_bwd
    etc_fwd = max(total_fwd_ms - accounted_fwd, 0.0)
    etc_bwd = max(total_bwd_ms - accounted_bwd, 0.0)

    return {
        "1_trunk_no_grad": {"fwd": trunk_wo_fwd, "bwd": 0.0},
        "2_trunk_with_grad": {"fwd": trunk_wg_fwd, "bwd": trunk_wg_bwd},
        "3_atom_dit": {"fwd": atom_fwd, "bwd": atom_bwd},
        "4_token_dit": {"fwd": token_fwd, "bwd": token_bwd},
        "5_etc": {"fwd": etc_fwd, "bwd": etc_bwd},
        "_total": {"fwd": total_fwd_ms, "bwd": total_bwd_ms},
        # Detail breakdown for the curious.
        "_detail": {
            "msa/nograd_fwd": fwd("msa/nograd"),
            "pairformer/nograd_fwd": fwd("pairformer/nograd"),
            "msa/grad_fwd": fwd("msa/grad"),
            "pairformer/grad_fwd": fwd("pairformer/grad"),
            "msa_bwd": bwd("msa"),
            "pairformer_bwd": bwd("pairformer"),
            "atom_dit_enc_fwd": fwd("atom_dit_enc/grad"),
            "atom_dit_dec_fwd": fwd("atom_dit_dec/grad"),
            "atom_dit_enc_bwd": bwd("atom_dit_enc"),
            "atom_dit_dec_bwd": bwd("atom_dit_dec"),
        },
    }


def print_summary(label: str, s: dict) -> None:
    click.echo(f"\n=== {label} ===")
    click.echo(f"  {'bucket':<22}{'fwd (ms)':>12}{'bwd (ms)':>12}{'sum (ms)':>12}")
    order = [
        "1_trunk_no_grad",
        "2_trunk_with_grad",
        "3_atom_dit",
        "4_token_dit",
        "5_etc",
    ]
    grand = s["_total"]["fwd"] + s["_total"]["bwd"]
    for k in order:
        f = s[k]["fwd"]
        b = s[k]["bwd"]
        click.echo(f"  {k:<22}{f:>12.2f}{b:>12.2f}{f + b:>12.2f}")
    tf = s["_total"]["fwd"]
    tb = s["_total"]["bwd"]
    click.echo(f"  {'-' * 56}")
    click.echo(f"  {'TOTAL (wall)':<22}{tf:>12.2f}{tb:>12.2f}{grand:>12.2f}")
    click.echo("  detail:")
    for k, v in s["_detail"].items():
        click.echo(f"    {k:<26}{v:>10.2f} ms")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/miniworld/config_exp_msa3_24_3_edm.yaml"),
    show_default=True,
)
@click.option("--model", "model_override", type=str, default="large_msa3_24_3", show_default=True)
@click.option("--n-recycle", "n_recycles", type=int, multiple=True, default=(4, 20), show_default=True)
@click.option(
    "--num-augment",
    "num_augments",
    type=int,
    multiple=True,
    default=(),
    help="num_augment value(s) to sweep. Empty -> train.num_augment.",
)
@click.option("--n-tokens", type=int, default=None, help="Override crop.max_tokens.")
@click.option("--n-atoms", type=int, default=None, help="Override crop.max_atoms.")
@click.option("--msa-depth", type=int, default=None, help="Override msa.max_msa_depth.")
@click.option("--n-iters", type=int, default=3, show_default=True, help="First iter is warmup.")
@click.option(
    "--freeze-trunk/--no-freeze-trunk",
    default=True,
    show_default=True,
    help="Run the trunk entirely under no_grad (matches real frozen-trunk training). "
    "Trunk-with-grad cost is then measured in a separate isolated pass.",
)
@click.option(
    "--override",
    "extra_overrides",
    type=str,
    multiple=True,
    default=(),
    help="Extra hydra override(s), e.g. model.trunk.pairformer.use_single=False.",
)
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(  # noqa: PLR0913, PLR0915
    config: Path,
    model_override: str,
    n_recycles: tuple[int, ...],
    num_augments: tuple[int, ...],
    n_tokens: int | None,
    n_atoms: int | None,
    msa_depth: int | None,
    n_iters: int,
    freeze_trunk: bool,
    extra_overrides: tuple[str, ...],
    json_out: Path | None,
) -> None:
    """Profile fwd/bwd time per module of the MiniWorld EDM Model."""
    from hydra import compose, initialize_config_dir

    if not torch.cuda.is_available():
        msg = "CUDA is required for this profile."
        raise RuntimeError(msg)
    device = torch.device("cuda")

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg_raw = compose(
            config_name=config.name,
            overrides=[f"model={model_override}", *extra_overrides],
        )
    cfg = Config.model_validate(cfg_raw)

    augs = list(num_augments) if num_augments else [cfg.train.num_augment]
    n_tok = n_tokens if n_tokens is not None else cfg.data.crop.max_tokens
    n_atm = n_atoms if n_atoms is not None else cfg.data.crop.max_atoms
    msa_d = msa_depth if msa_depth is not None else cfg.data.msa.max_msa_depth

    click.echo(
        f"Profile: model={model_override} num_augments={augs} "
        f"tokens={n_tok} atoms={n_atm} msa_depth={msa_d} "
        f"recycles={list(n_recycles)} overrides={list(extra_overrides)} "
        f"gpu={torch.cuda.get_device_name(0)}",
    )

    model = Model(cfg.model).to(device)
    model.train()

    batch = _build_precompile_batch(
        device=device,
        msa_depth=msa_d,
        n_tokens=n_tok,
        n_atoms=n_atm,
        n_templates=TemplateConfig().n_templates,
        num_res_class=cfg.model.shared.num_res_class,
    )

    # Build the real EDM diffuser so the noised inputs (x_input, x_mask, t_emb)
    # have exactly the shapes training feeds into Model.forward.
    scheduler = EDMScheduler(cfg.diffuser.scheduler)
    diffuser = EuclideanDiffuser(
        config=EuclideanDiffuser.EuclideanConfig(seed=cfg.diffuser.seed),
        scheduler=scheduler,
    )

    timers = Timers()
    dm = model.diffusion_module
    restores = [
        wrap_module(model.msa_module, "msa", timers),
        wrap_module(model.pairformer_blocks, "pairformer", timers),
        wrap_module(dm.atom_attention_encoder.atom_transformer, "atom_dit_enc", timers),
        wrap_module(dm.atom_attention_decoder.atom_transformer, "atom_dit_dec", timers),
        wrap_module(dm.diffusion_transformer, "token_dit", timers),
    ]

    results: dict[str, dict] = {}

    # --- Optionally run the trunk under no_grad (matches frozen-trunk training).
    orig_cond = model.condition_forward
    if freeze_trunk:
        def cond_nograd(*a, **k):  # noqa: ANN002, ANN003
            with torch.no_grad():
                return orig_cond(*a, **k)
        model.condition_forward = cond_nograd  # type: ignore[method-assign]

    def one_iter(n_recycle: int, x_t, x_mask, t_emb) -> tuple[dict, float]:  # noqa: ANN001
        timers.reset()
        model._forced_n_recycle = n_recycle  # noqa: SLF001
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        fwd_s = torch.cuda.Event(enable_timing=True)
        fwd_e = torch.cuda.Event(enable_timing=True)
        bwd_s = torch.cuda.Event(enable_timing=True)
        bwd_e = torch.cuda.Event(enable_timing=True)

        fwd_s.record()
        atom_pos_update, distogram_logit = model.forward(
            msa=batch.msa,
            template=batch.template,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
            x_t=x_t,
            x_mask=x_mask,
            t_emb=t_emb,
        )
        fwd_e.record()

        loss = atom_pos_update.float().pow(2).mean() + distogram_logit.float().pow(2).mean()
        bwd_s.record()
        loss.backward()
        bwd_e.record()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        total_fwd = fwd_s.elapsed_time(fwd_e)
        total_bwd = bwd_s.elapsed_time(bwd_e)
        return summarize(timers, total_fwd, total_bwd), peak_gb

    # --- Isolated trunk-with-grad: ONE recycle cycle, no diffusion module.
    # Gives the per-cycle fwd+bwd cost the trunk *would* add if it were not
    # frozen. Fits memory easily (no 48-augment diffusion graph).
    def measure_trunk_with_grad() -> dict:
        timers.reset()
        model._forced_n_recycle = 1  # noqa: SLF001  -- single grad cycle
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        fs = torch.cuda.Event(enable_timing=True)
        fe = torch.cuda.Event(enable_timing=True)
        bs = torch.cuda.Event(enable_timing=True)
        be = torch.cuda.Event(enable_timing=True)
        fs.record()
        _tsi, ts, tp, disto = orig_cond(  # grad ON (bypass the no_grad patch)
            batch.msa, batch.template, batch.reference,
            batch.scheme, batch.sequence, batch.structure,
        )
        fe.record()
        loss = ts.float().pow(2).mean() + tp.float().pow(2).mean() + disto.float().pow(2).mean()
        bs.record()
        loss.backward()
        be.record()
        torch.cuda.synchronize()
        return {
            "fwd": timers.fwd_ms("msa/grad") + timers.fwd_ms("pairformer/grad"),
            "bwd": timers.bwd_ms("msa") + timers.bwd_ms("pairformer"),
            "wall_fwd": fs.elapsed_time(fe),
            "wall_bwd": bs.elapsed_time(be),
        }

    try:
        # Trunk-with-grad cost per cycle (recycle-independent: 1 grad cycle).
        trunk_wg = None
        try:
            for i in range(n_iters):
                trunk_wg = measure_trunk_with_grad()
            click.echo(
                f"\n=== trunk WITH grad (per 1 cycle) ===\n"
                f"  msa+pairformer fwd={trunk_wg['fwd']:.2f} ms  "
                f"bwd={trunk_wg['bwd']:.2f} ms  "
                f"(wall fwd={trunk_wg['wall_fwd']:.2f} bwd={trunk_wg['wall_bwd']:.2f})",
            )
            results["trunk_with_grad_per_cycle"] = trunk_wg
        except torch.OutOfMemoryError:
            click.echo("\n!!! OOM measuring isolated trunk-with-grad")
            torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        for n_aug in augs:
            for n_recycle in n_recycles:
                key = f"aug{n_aug}_rec{n_recycle}"
                _x0, x_t, x_mask, t_emb, _sigma = diffuser.sample(
                    x0=batch.structure.atom_pos,
                    mask=batch.structure.atom_pos_mask,
                    num_augment=n_aug,
                )
                measured: list[dict] = []
                peak = 0.0
                try:
                    for i in range(n_iters):
                        s, peak = one_iter(n_recycle, x_t, x_mask, t_emb)
                        label = "warmup" if i == 0 else f"iter {i}"
                        print_summary(f"[{key}] {label} (peak {peak:.1f} GB)", s)
                        if i > 0:
                            measured.append(s)
                except torch.OutOfMemoryError:
                    click.echo(f"\n!!! OOM at {key} -- skipping")
                    model.zero_grad(set_to_none=True)
                    del x_t, x_mask, t_emb
                    torch.cuda.empty_cache()
                    continue
                if measured:
                    mean = _mean_summaries(measured)
                    # Fill category-2 from the isolated trunk-with-grad measurement.
                    if trunk_wg is not None:
                        mean["2_trunk_with_grad"] = {
                            "fwd": trunk_wg["fwd"],
                            "bwd": trunk_wg["bwd"],
                        }
                    mean["_peak_gb"] = peak
                    print_summary(f"[{key}] MEAN", mean)
                    results[key] = mean
                del x_t, x_mask, t_emb
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
    finally:
        model.condition_forward = orig_cond  # type: ignore[method-assign]
        for r in restores:
            r()

    if json_out is not None:
        json_out.write_text(json.dumps(results, indent=2))
        click.echo(f"\nWrote {json_out}")


def _mean_summaries(measured: list[dict]) -> dict:
    keys = ["1_trunk_no_grad", "2_trunk_with_grad", "3_atom_dit", "4_token_dit", "5_etc", "_total"]
    out: dict = {}
    n = len(measured)
    for k in keys:
        out[k] = {
            "fwd": sum(m[k]["fwd"] for m in measured) / n,
            "bwd": sum(m[k]["bwd"] for m in measured) / n,
        }
    detail_keys = measured[0]["_detail"].keys()
    out["_detail"] = {dk: sum(m["_detail"][dk] for m in measured) / n for dk in detail_keys}
    return out


if __name__ == "__main__":
    main()
