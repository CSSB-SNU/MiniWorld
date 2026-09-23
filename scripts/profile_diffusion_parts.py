"""Where does the phase-2 diffusion step spend its time: atom DiT or token DiT?

Splits ``DiffusionModel.diffusion_forward`` into its four parts and times each one in
isolation on the tensors it really receives:

* ``diffusion_conditioning``   RelPos + pair/single transitions + time embedding (per pair!)
* ``atom_attention_encoder``   ESMFold2 SWA atom DiT, 3 blocks, atoms x num_augment
* ``diffusion_transformer``    AF3 token DiT, 24 blocks, tokens x num_augment, pair-biased
* ``atom_attention_decoder``   ESMFold2 SWA atom DiT, 3 blocks, atoms x num_augment

Method: one grad-enabled full forward with forward hooks that capture every part's exact
(args, kwargs). Each part is then timed alone -- forward only, and forward+backward with a
squared-output loss -- on detached copies of those inputs (float inputs get requires_grad so
the backward covers input gradients too, as in the real step). The full
``diffusion_forward`` fwd+bwd is timed as the total; ``remainder`` is total minus the parts
(the glue: to_token_single_trunk, add_single_token_cond, ln_token_single_rep, autograd
bookkeeping). Eager and per-part ``torch.compile(dynamic=False)`` are both reported.

Needs a GPU -- run as a batch job. One config per process (shapes differ per phase).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.setrecursionlimit(30000)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_diffusion_train import Config  # noqa: E402

from miniworld.data.dataloader.dataloader import BioMolData  # noqa: E402
from miniworld.models.diffusion import Client  # noqa: E402

PARTS = ("diffusion_conditioning", "atom_attention_encoder",
         "diffusion_transformer", "atom_attention_decoder")


def timed(fn, n_warm: int, n_rep: int) -> tuple[float, float]:
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def _detach_tree(obj, grad: bool):
    """Detached copy of a (possibly nested / dataclass) argument; float tensors get grad."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach().clone()
        if grad and t.is_floating_point():
            t.requires_grad_(True)
        return t
    # Tensors nested in containers are carries, not activations: the SWA decoder's
    # ``attention_params`` holds the RoPE cos/sin tables, which the engine requires to be
    # constants ("positional cos/sin tables must be constants"). Only top-level tensor
    # arguments are activations that receive gradients in the real step.
    if isinstance(obj, (tuple, list)):
        return type(obj)(_detach_tree(o, False) for o in obj)
    if isinstance(obj, dict):
        return {k: _detach_tree(v, False) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{
            f.name: _detach_tree(getattr(obj, f.name), False) for f in dataclasses.fields(obj)
        })
    return obj


def _sq_loss(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.float().pow(2).mean() if out.is_floating_point() and out.requires_grad else None
    parts = [_sq_loss(o) for o in (out if isinstance(out, (tuple, list)) else [])]
    parts = [p for p in parts if p is not None]
    return torch.stack(parts).sum() if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rep", type=int, default=7)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fabric-precision", default="32-true",
                    help="Fabric precision. '32-true' = what the phase-2 trainer runs (no autocast; "
                         "fp32 DiT params run TF32 GEMMs, a bf16 DiT runs bf16). 'bf16-mixed' = the "
                         "autocast setting the earlier profiles used.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Hydra overrides, e.g. data.train_db.catalog_cache_path=<fingerprint-matching "
                         "arrow>; without it two array tasks race to rebuild a stale shared catalog")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=cfg_path.name, overrides=args.override))

    from lightning import Fabric
    torch.set_float32_matmul_precision("medium")  # trainer sets this too
    fabric = Fabric(accelerator="cuda", devices=1, precision=args.fabric_precision)
    fabric.launch()
    client = Client(Client.Config(train=cfg.train, model=cfg.model,
                                  diffuser=cfg.diffuser, loss=cfg.loss))
    client.config.train.param_policy.enabled = False
    client.setup(fabric=fabric)
    if args.ckpt:
        client.load_state_dict(torch.load(args.ckpt, map_location="cpu"),
                               model_only=True, strict=False)
    client.model.train()
    raw = getattr(client.model, "module", client.model)
    dm = raw.diffusion_module

    data = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
    loader = data.create_ddp_dataloader(
        world_size=1, rank=0, seed=0, drop_last=False, batch_size=1, num_workers=4,
        num_samples_per_rank=4, shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
        bucket_template_multiple=4)
    data.set_epoch(0)
    it = iter(loader)
    batch = next(it).to(device=client.device)
    del it, loader
    aug = cfg.train.num_augment
    shape = dict(tokens=int(batch.token_length), atoms=int(batch.atom_length),
                 msa=int(batch.msa_depth), num_augment=aug)
    print(f"config={cfg_path.name} batch={shape}", flush=True)

    raw._forced_n_recycle = 1  # noqa: SLF001
    with torch.no_grad():
        tsi, tpt = raw.condition_forward(batch.msa, batch.reference, batch.scheme,
                                         batch.sequence, batch.structure, batch.template)
    x0, x_input, x_mask, t_emb, _ = client.diffuser.sample(
        batch.structure.atom_pos, num_augment=aug, mask=batch.structure.atom_pos_mask)

    # ---- capture each part's real inputs with one grad-enabled full forward --------
    captured: dict[str, tuple] = {}
    hooks = []
    for name in PARTS:
        mod = getattr(dm, name)
        hooks.append(mod.register_forward_hook(
            lambda m, a, kw, out, _n=name: captured.__setitem__(_n, (a, kw)), with_kwargs=True))
    raw.zero_grad(set_to_none=True)
    out = raw.diffusion_forward(batch.reference, batch.scheme, batch.structure,
                                x_input, x_mask, t_emb, tsi, tpt)
    out.float().pow(2).mean().backward()
    for h in hooks:
        h.remove()
    assert set(captured) == set(PARTS), set(captured)
    raw.zero_grad(set_to_none=True)

    # a Linear weight, not the first parameter: that is an fp32-pinned norm gamma and would
    # report fp32 for a bf16 module
    dit_dtype = str(next(p for n, p in dm.diffusion_transformer.named_parameters()
                         if n.endswith("to_value.weight") or n.endswith("to_out.weight")).dtype)
    print(f"fabric precision={args.fabric_precision}  token DiT param dtype={dit_dtype}  "
          f"token_dit_kind={getattr(dm, 'token_dit_kind', 'augmented')}", flush=True)
    results = {"config": cfg_path.name, "batch": shape, "parts": {}, "total": {},
               "fabric_precision": args.fabric_precision, "dit_param_dtype": dit_dtype,
               "token_dit_kind": getattr(dm, "token_dit_kind", "augmented")}

    def full_fwd_bwd() -> None:
        raw.zero_grad(set_to_none=True)
        o = raw.diffusion_forward(batch.reference, batch.scheme, batch.structure,
                                  x_input, x_mask, t_emb, tsi, tpt)
        o.float().pow(2).mean().backward()

    for mode in ("eager", "compiled"):
        print(f"\n===== {mode} =====", flush=True)
        if mode == "compiled":
            torch._dynamo.reset()
            torch._dynamo.config.cache_size_limit = 64
            torch._dynamo.config.accumulated_cache_size_limit = 256
        res = {}
        for name in PARTS:
            mod = getattr(dm, name)
            fn = torch.compile(mod, dynamic=False) if mode == "compiled" else mod
            a, kw = captured[name]
            a_g, kw_g = _detach_tree(a, True), _detach_tree(kw, True)
            a_n, kw_n = _detach_tree(a, False), _detach_tree(kw, False)

            def fwd_only(fn=fn, a=a_n, kw=kw_n) -> None:
                with torch.no_grad():
                    fn(*a, **kw)

            def fwd_bwd(fn=fn, a=a_g, kw=kw_g) -> None:
                mod.zero_grad(set_to_none=True)
                o = fn(*a, **kw)
                loss = _sq_loss(o)
                loss.backward()
            f, fsd = timed(fwd_only, args.warm, args.rep)
            fb, fbsd = timed(fwd_bwd, args.warm, args.rep)
            res[name] = dict(fwd=f, fwd_std=fsd, fwd_bwd=fb, fwd_bwd_std=fbsd)
            print(f"{name:26s} fwd {f:7.3f}s   fwd+bwd {fb:7.3f}s  (std {fbsd:.3f})", flush=True)
        if mode == "compiled":
            raw.diffusion_forward_compiled = torch.compile(raw.diffusion_forward, dynamic=False)

            def full_fwd_bwd_c() -> None:
                raw.zero_grad(set_to_none=True)
                o = raw.diffusion_forward_compiled(batch.reference, batch.scheme, batch.structure,
                                                   x_input, x_mask, t_emb, tsi, tpt)
                o.float().pow(2).mean().backward()
            tot, tsd = timed(full_fwd_bwd_c, args.warm, args.rep)
        else:
            tot, tsd = timed(full_fwd_bwd, args.warm, args.rep)
        parts_sum = sum(r["fwd_bwd"] for r in res.values())
        print(f"{'FULL diffusion_forward':26s} fwd+bwd {tot:7.3f}s  (std {tsd:.3f})   "
              f"parts sum {parts_sum:.3f}s   remainder {tot - parts_sum:+.3f}s", flush=True)
        print(f"\n{'part':26s}{'fwd+bwd (s)':>13}{'% of total':>12}")
        for name, r in res.items():
            print(f"{name:26s}{r['fwd_bwd']:>13.3f}{100 * r['fwd_bwd'] / tot:>11.1f}%")
        print(f"{'remainder (glue)':26s}{tot - parts_sum:>13.3f}{100 * (tot - parts_sum) / tot:>11.1f}%")
        atom = res["atom_attention_encoder"]["fwd_bwd"] + res["atom_attention_decoder"]["fwd_bwd"]
        tok = res["diffusion_transformer"]["fwd_bwd"]
        print(f"\natom DiT (enc+dec) {atom:.3f}s = {100 * atom / tot:.1f}%   "
              f"token DiT {tok:.3f}s = {100 * tok / tot:.1f}%   "
              f"conditioning {res['diffusion_conditioning']['fwd_bwd']:.3f}s = "
              f"{100 * res['diffusion_conditioning']['fwd_bwd'] / tot:.1f}%", flush=True)
        results["parts"][mode] = res
        results["total"][mode] = dict(fwd_bwd=tot, std=tsd, parts_sum=parts_sum)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
