"""GPU smoke test of model.trunk.restype_msa_only on the real v2.4.0 trunk: one GPU, random weights, a synthetic batch, no data.

    sbatch submit/check_restype_routing.sh

Passes (exit status 0) when, with the flag on,
  * changing only the residue types (sequence.token_type) leaves the pair init unchanged, and changes the MSA module's input
    single in the restype channels only;
  * conditioning_single() (what phase 2 gives the diffusion module) is unchanged as well;
  * a backward pass from random linear read-outs of the pair init and of the MSA module's single projection leaves exactly zero
    gradient on the restype inputs of the pair-init projections, and a nonzero one on the restype inputs of the MSA projection;
and, with the flag off, the same change does reach the pair init (so the first check can fail at all).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/miniworld/phase1a_distogram_medium_v240.yaml"


def main() -> int:
    sys.path.insert(0, str(REPO / "scripts"))
    from hydra import compose, initialize_config_dir
    from run_miniworld_distogram_train import Config, _build_precompile_batch

    from miniworld.models.distogram_only import MiniSWAModel
    from miniworld.training.engine_backend import (
        apply_run_options,
        configure_engine_backend,
        configure_fused_msa_train,
    )
    from miniworld.training.precision import apply_precision, model_autocast

    with initialize_config_dir(str(CONFIG.parent), version_base=None):
        raw = compose(
            config_name=CONFIG.name, overrides=["+model.trunk.restype_msa_only=true"]
        )
    cfg = Config.model_validate(raw)
    apply_run_options(cfg.train)
    configure_engine_backend(cfg.train.engine_backend)
    configure_fused_msa_train(bool(cfg.train.fused_msa_train))

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    model = apply_precision(MiniSWAModel(cfg.model), cfg.train.precision).to(device)
    shared = cfg.model.shared
    lo, hi = (
        shared.d_single,
        shared.d_single + shared.num_res_class,
    )  # restype channels of the input single

    batch = _build_precompile_batch(
        device=device,
        msa_depth=1024,
        n_tokens=cfg.data.crop.max_tokens,
        n_atoms=cfg.data.crop.max_atoms,
        n_templates=4,
        num_res_class=shared.num_res_class,
    )
    other = copy.deepcopy(batch)
    other.sequence.token_type = (
        batch.sequence.token_type + 7
    ) % 20  # residue types only; MSA, atoms, templates unchanged

    def embed(b):
        model.eval()
        with torch.no_grad(), model_autocast(model):
            pair, single, *_ = model._embed(
                b.msa, b.reference, b.scheme, b.sequence, b.structure
            )
        return pair.float(), single.float()

    checks: dict[str, bool] = {}
    pair_a, single_a = embed(batch)
    pair_b, single_b = embed(other)
    model.config.trunk.restype_msa_only = False
    pair_a_off, _ = embed(batch)
    pair_b_off, _ = embed(other)
    model.config.trunk.restype_msa_only = True

    on = float((pair_a - pair_b).abs().max())
    off = float((pair_a_off - pair_b_off).abs().max())
    print(
        f"pair init change from new residue types: flag on {on:.3g}, flag off {off:.3g}"
    )
    checks["flag off: residue types reach the pair init"] = off > 0
    checks["flag on: pair init unchanged"] = on == 0 or on < 1e-3 * off
    diff = (single_a - single_b).abs()
    checks["flag on: MSA single changes in the restype channels"] = (
        float(diff[..., lo:hi].max()) > 0
    )
    checks["flag on: MSA single unchanged elsewhere"] = (
        float(diff[..., :lo].max()) == 0 and float(diff[..., hi:].max()) == 0
    )
    checks["flag on: conditioning_single unchanged"] = torch.equal(
        model.conditioning_single(single_a), model.conditioning_single(single_b)
    )

    # Gradients through the embedding itself: a random linear read-out of the pair init and of the MSA module's single projection.
    # A loss on the full model's logits cannot test this at initialization: the distogram head's output layer starts at zero, so
    # every gradient upstream of it is exactly zero (2026-11-04, job 42233: the routing passed, this part reported all zeros).
    model.train()
    model.zero_grad(set_to_none=True)
    with model_autocast(model):
        pair, single, *_ = model._embed(
            batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure
        )
        msa_in = model.msa_module.single_to_msa(single)
    gen = torch.Generator(device=device).manual_seed(0)
    readout = (
        pair.float() * torch.randn(pair.shape, generator=gen, device=device)
    ).sum()
    readout = (
        readout
        + (
            msa_in.float() * torch.randn(msa_in.shape, generator=gen, device=device)
        ).sum()
    )
    readout.backward()

    def restype_and_rest(weight: torch.Tensor) -> tuple[float, float]:
        grad = weight.grad
        grad = (
            grad
            if grad.shape[-1] == shared.d_single_token_input
            else grad.transpose(0, 1)
        )
        return float(grad[..., lo:hi].abs().max()), float(grad[..., :lo].abs().max())

    embedder = model.input_feature_embedder
    for name in ("to_token_pair_left", "to_token_pair_right"):
        restype, rest = restype_and_rest(getattr(embedder, name).weight)
        print(
            f"input_feature_embedder.{name}: max |grad| restype inputs {restype:.3g}, atom inputs {rest:.3g}"
        )
        checks[f"{name}: zero gradient on restype inputs"] = restype == 0
        checks[f"{name}: nonzero gradient on atom inputs"] = rest > 0
    restype, _ = restype_and_rest(model.msa_module.single_to_msa.weight)
    print(f"msa_module.single_to_msa: max |grad| restype inputs {restype:.3g}")
    checks["msa_module.single_to_msa: nonzero gradient on restype inputs"] = restype > 0

    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("restype routing check:", "PASSED" if all(checks.values()) else "FAILED")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    # Exit 0 = routing verified, 1 = a routing check FAILED, 2 = the check itself could not run (an error, not a verdict).
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("restype routing check: ERROR (no verdict)")
        sys.exit(2)
