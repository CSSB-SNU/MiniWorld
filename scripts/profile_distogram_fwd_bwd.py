"""Per-module FORWARD and BACKWARD timing for the distogram-only model.

profile_distogram_only.py times forward per-module but lumps the whole backward
into one wall number. This script inserts identity autograd "probe" nodes on
each top-level submodule's input/output tensors, so a CUDA event is recorded
when the forward runs AND when gradient flows back through it -> real per-module
forward and backward times.

Why probes (not module hooks): the model calls ``pairformer_blocks.forward(...)``
directly (bypassing __call__/hooks), and ``input_feature_embedder`` gets inputs
that don't require grad (so full_backward_hook misfires). Graph-inserted probes
handle both correctly.

Usage:
    pixi run -e cu128 python scripts/profile_distogram_fwd_bwd.py \
        --model medium_distogram_8b_norecycle --n-recycle 1 --n-iters 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_distogram_train import Config, _build_precompile_batch

from miniworld.configs.data import TemplateConfig
from miniworld.models.distogram_only import Model

torch.set_float32_matmul_precision("medium")

TARGETS = [
    "input_feature_embedder",
    "msa_module",
    "pairformer_blocks",
    "add_pair_recycle",
    "distogram_head",
]


def _event() -> torch.cuda.Event:
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


class _ProbeOut(torch.autograd.Function):
    """Identity on a module OUTPUT: fwd records fwd-end, bwd records bwd-start."""

    @staticmethod
    def forward(ctx, x, slot, store):  # noqa: ANN001
        store[f"{slot}_fwd_end"] = _event()
        ctx.slot, ctx.store = slot, store
        return x

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001
        ctx.store[f"{ctx.slot}_bwd_start"] = _event()
        return g, None, None


class _ProbeIn(torch.autograd.Function):
    """Identity on a module INPUT: bwd records bwd-end (grad reached input)."""

    @staticmethod
    def forward(ctx, x, slot, store):  # noqa: ANN001
        ctx.slot, ctx.store = slot, store
        return x

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001
        ctx.store[f"{ctx.slot}_bwd_end"] = _event()
        return g, None, None


class _Wrap(nn.Module):
    def __init__(self, mod: nn.Module, name: str, store: dict):  # noqa: ANN001
        super().__init__()
        self.mod = mod
        self._name = name
        self._store = store

    def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003
        args = list(args)
        for i, t in enumerate(args):
            if torch.is_tensor(t) and t.requires_grad and t.is_floating_point():
                args[i] = _ProbeIn.apply(t, self._name, self._store)
                break
        self._store[f"{self._name}_fwd_start"] = _event()
        out = self.mod(*args, **kwargs)
        if torch.is_tensor(out):
            return _ProbeOut.apply(out, self._name, self._store)
        if isinstance(out, (tuple, list)):
            out = list(out)
            for i, t in enumerate(out):
                if torch.is_tensor(t) and t.is_floating_point():
                    out[i] = _ProbeOut.apply(t, self._name, self._store)
                    break
            return tuple(out)
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/miniworld/config_distogram.yaml")
    p.add_argument("--model", dest="model_override", default="medium_distogram_8b_norecycle")
    p.add_argument("--n-recycle", type=int, default=1)
    p.add_argument("--n-iters", type=int, default=5, help="1 warmup + rest measured")
    args = p.parse_args()

    if not torch.cuda.is_available():
        msg = "CUDA required"
        raise RuntimeError(msg)
    device = torch.device("cuda")

    cfg_path = Path(args.config)
    with initialize_config_dir(str(cfg_path.parent.absolute()), version_base=None):
        cfg_raw = compose(
            config_name=cfg_path.name,
            overrides=[f"model={args.model_override}", "train=distogram_v0.1"],
        )
    cfg = Config.model_validate(cfg_raw)

    print(f"Profile fwd+bwd: model={args.model_override} n_recycle={args.n_recycle}")
    print(
        f"Shape: msa={cfg.data.msa.max_msa_depth} "
        f"tokens={cfg.data.crop.max_tokens} atoms={cfg.data.crop.max_atoms}",
    )

    model = Model(cfg.model).to(device)
    model.train()
    model._forced_n_recycle = args.n_recycle  # noqa: SLF001

    store: dict[str, torch.cuda.Event] = {}
    for name in TARGETS:
        setattr(model, name, _Wrap(getattr(model, name), name, store))

    n_templates = TemplateConfig().n_templates
    batch = _build_precompile_batch(
        device=device,
        msa_depth=cfg.data.msa.max_msa_depth,
        n_tokens=cfg.data.crop.max_tokens,
        n_atoms=cfg.data.crop.max_atoms,
        n_templates=n_templates,
        num_res_class=cfg.model.shared.num_res_class,
    )

    fwd_acc: dict[str, float] = defaultdict(float)
    bwd_acc: dict[str, float] = defaultdict(float)

    def one_iter(measure: bool) -> None:  # noqa: FBT001
        store.clear()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        logits = model(
            batch.msa,
            batch.reference,
            batch.scheme,
            batch.sequence,
            batch.structure,
        )
        logits.float().sum().backward()
        global_bwd_end = _event()
        torch.cuda.synchronize()
        if not measure:
            return
        for name in TARGETS:
            fs, fe = store.get(f"{name}_fwd_start"), store.get(f"{name}_fwd_end")
            if fs is not None and fe is not None:
                fwd_acc[name] += fs.elapsed_time(fe)
            bs = store.get(f"{name}_bwd_start")
            be = store.get(f"{name}_bwd_end", global_bwd_end)
            if bs is not None:
                bwd_acc[name] += bs.elapsed_time(be)

    one_iter(measure=False)  # warmup
    n_measured = max(1, args.n_iters - 1)
    for _ in range(n_measured):
        one_iter(measure=True)

    print(f"\n=== per-module fwd / bwd (mean over {n_measured} iters, ms) ===")
    print(f"  {'module':28s} {'fwd':>9s} {'bwd':>9s} {'fwd+bwd':>9s}")
    tot_f = tot_b = 0.0
    for name in TARGETS:
        f = fwd_acc[name] / n_measured
        b = bwd_acc[name] / n_measured
        tot_f += f
        tot_b += b
        print(f"  {name:28s} {f:9.2f} {b:9.2f} {f + b:9.2f}")
    print(f"  {'-' * 56}")
    print(f"  {'SUM':28s} {tot_f:9.2f} {tot_b:9.2f} {tot_f + tot_b:9.2f}")
    print(
        "  note: input_feature_embedder bwd has no grad-requiring input, so its\n"
        "  bwd is measured to end-of-backward (includes tiny add_pair_recycle bwd).",
    )


if __name__ == "__main__":
    main()
