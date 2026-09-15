"""Where does a phase-2b training step spend its time: frozen trunk or diffusion module?

Times the two halves directly and separately:

* `condition_forward` under no_grad -- the frozen trunk, whose cost scales with the
  recycle count (training samples n_recycle ~ U[1, n_recycle_max=4], mean 2.5).
* `diffusion_forward` forward+backward -- the trainable half, whose cost scales with
  num_augment (48) and is independent of the recycle count.

With `--compile` (default) each half is wrapped in its own `torch.compile(dynamic=False)`
rather than going through `client.model.compile()` as training does -- driving the whole
compiled model from a timing loop sends dynamo into a RecursionError while it traces
Lightning's Fabric wrapper. Graph boundaries therefore differ slightly from training's
whole-model compile, but the per-half numbers are compiled ones. `--no-compile` is eager.

Needs a GPU -- run it as a batch job.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.setrecursionlimit(30000)  # dynamo traces deeply through the diffuser + loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_diffusion_train import Config  # noqa: E402

from miniworld.data.dataloader.dataloader import BioMolData  # noqa: E402
from miniworld.models.diffusion import Client  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--recycles", default="1,2,3,4,6,8,10")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rep", type=int, default=5)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    # split     : each half its own torch.compile(dynamic=False)
    # cudagraph : train.trunk_compile_mode -- frozen trunk under inductor
    #             cudagraph-trees, diffusion module under plain compile
    # whole     : client.model.compile(dynamic=False) exactly as training does.
    #             The two halves cannot be timed apart here (one dynamo entry at
    #             DiffusionModel.forward), so only the FULL step is reported --
    #             compare it against split's trunk+diffusion sum.
    ap.add_argument("--mode", choices=("split", "cudagraph", "whole"), default="split")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=cfg_path.name))

    from lightning import Fabric
    fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()

    client = Client(Client.Config(train=cfg.train, model=cfg.model,
                                  diffuser=cfg.diffuser, loss=cfg.loss))
    client.config.train.param_policy.enabled = False
    client.setup(fabric=fabric)
    if args.ckpt:
        client.load_state_dict(torch.load(args.ckpt, map_location="cpu"),
                               model_only=True, strict=False)
    client.model.train()

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
    del it, loader  # the batch is on the GPU now; shut the workers down cleanly
    print(f"batch: tokens={int(batch.token_length)} atoms={int(batch.atom_length)} "
          f"msa={int(batch.msa_depth)}  num_augment={cfg.train.num_augment}")

    raw = getattr(client.model, "module", client.model)

    if args.compile:
        # one graph variant per recycle count, plus the diffusion graph
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256

    if not args.compile:
        cond_fn, diff_fn = raw.condition_forward, raw.diffusion_forward
        print("eager", flush=True)
    elif args.mode == "cudagraph":
        raw.enable_trunk_cudagraph("reduce-overhead")
        cond_fn = raw.condition_forward  # dispatches to the compiled trunk
        diff_fn = torch.compile(raw.diffusion_forward, dynamic=False)
        print("compiled: trunk reduce-overhead (cudagraph) + "
              "diffusion_forward dynamic=False", flush=True)
    elif args.mode == "whole":
        raw.compile(dynamic=False)  # nn.Module.compile -- what training does
        cond_fn, diff_fn = raw.condition_forward, raw.diffusion_forward
        print("compiled: WHOLE model (dynamic=False), full-step timing only",
              flush=True)
    else:
        cond_fn = torch.compile(raw.condition_forward, dynamic=False)
        diff_fn = torch.compile(raw.diffusion_forward, dynamic=False)
        print("compiled: condition_forward + diffusion_forward (dynamic=False)",
              flush=True)

    if args.mode == "whole":
        aug = cfg.train.num_augment
        x0, x_input, x_mask, t_emb, _ = client.diffuser.sample(
            batch.structure.atom_pos, num_augment=aug,
            mask=batch.structure.atom_pos_mask)

        print(f"\n{'recycles':>9}{'full step (s)':>15}{'std':>8}")
        print("-" * 32)
        for k in [int(x) for x in args.recycles.split(",")]:
            raw._forced_n_recycle = k  # noqa: SLF001

            def run_full() -> None:
                raw.zero_grad(set_to_none=True)
                out = raw(batch.msa, batch.template, batch.reference, batch.scheme,
                          batch.sequence, batch.structure, x_input, x_mask, t_emb)
                out.float().pow(2).mean().backward()
            m, sd = timed(run_full, args.warm, args.rep)
            print(f"{k:>9}{m:>15.3f}{sd:>8.3f}")
        return

    print(f"\n{'recycles':>9}{'trunk (s)':>12}{'std':>8}{'per recycle':>13}")
    print("-" * 42)
    trunk = {}
    for k in [int(x) for x in args.recycles.split(",")]:
        raw._forced_n_recycle = k  # noqa: SLF001

        def run_trunk() -> None:
            with torch.no_grad():
                cond_fn(batch.msa, batch.reference, batch.scheme,
                        batch.sequence, batch.structure, batch.template)
        m, sd = timed(run_trunk, args.warm, args.rep)
        trunk[k] = m
        print(f"{k:>9}{m:>12.3f}{sd:>8.3f}{m / k:>13.3f}")

        if args.mode == "cudagraph" and k == min(trunk):
            # Addresses alone do not establish graph coverage: eager tail ops,
            # clones and graph-tree lifetime tracking can change these addresses.
            # Use audit_phase2_cudagraph.py for actual launch/capture counts.
            with torch.no_grad():
                a = cond_fn(batch.msa, batch.reference, batch.scheme,
                            batch.sequence, batch.structure, batch.template)
                pa = [t.data_ptr() for t in a]
                b = cond_fn(batch.msa, batch.reference, batch.scheme,
                            batch.sequence, batch.structure, batch.template)
                pb = [t.data_ptr() for t in b]
            static = pa == pb
            print(f"  [cudagraph probe] output ptrs stable={static} "
                  "(address observation only; graph activity not determined)")
            try:
                from torch._inductor import cudagraph_trees
                mgr = cudagraph_trees.get_manager(0, create_if_none_exists=False)
                print(f"  [cudagraph probe] tree manager={mgr is not None}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [cudagraph probe] manager lookup failed: {exc}")

    raw._forced_n_recycle = 1  # noqa: SLF001
    with torch.no_grad():
        tsi, tpt = cond_fn(batch.msa, batch.reference, batch.scheme,
                           batch.sequence, batch.structure, batch.template)
    if args.mode == "cudagraph":
        # cudagraph-managed buffers: a later replay would overwrite them
        tsi, tpt = tsi.clone(), tpt.clone()
    aug = cfg.train.num_augment
    x0, x_input, x_mask, t_emb, _ = client.diffuser.sample(
        batch.structure.atom_pos, num_augment=aug, mask=batch.structure.atom_pos_mask)

    def run_diff() -> None:
        raw.zero_grad(set_to_none=True)
        out = diff_fn(batch.reference, batch.scheme, batch.structure,
                      x_input, x_mask, t_emb, tsi, tpt)
        out.float().pow(2).mean().backward()
    d, dsd = timed(run_diff, args.warm, args.rep)
    print(f"\ndiffusion fwd+bwd (num_augment={aug}): {d:.3f} s  (std {dsd:.3f})")

    print(f"\n{'recycles':>9}{'trunk':>9}{'diffusion':>11}{'step':>9}{'trunk %':>9}")
    print("-" * 48)
    for k, t in trunk.items():
        print(f"{k:>9}{t:>9.3f}{d:>11.3f}{t + d:>9.3f}{100 * t / (t + d):>8.1f}%")
    eff = (cfg.model.trunk.n_recycle_max + 1) / 2
    # trunk(k) = a + b*k: fit from the measured points so a single-recycle run is
    # not required (each array task measures one recycle count).
    ks = sorted(trunk)
    if len(ks) > 1:
        b = (trunk[ks[-1]] - trunk[ks[0]]) / (ks[-1] - ks[0])
        a = trunk[ks[0]] - b * ks[0]
    else:
        b, a = trunk[ks[0]] / ks[0], 0.0
    t = a + b * eff
    print(f"\ntraining draws n_recycle ~ U[1,{cfg.model.trunk.n_recycle_max}] (mean {eff}):"
          f" trunk {t:.3f}s + diffusion {d:.3f}s = {t + d:.3f}s,"
          f" trunk {100 * t / (t + d):.1f}%")


if __name__ == "__main__":
    main()
