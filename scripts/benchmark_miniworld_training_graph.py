"""Full-size Phase-1 training CUDA-graph A/B on identical cached real batches.

Run arms in fresh processes. Within each pair only manual CUDA graph differs;
--compile enables Inductor in both arms. Includes Adam, clipping, scheduler and EMA in step
wall time. Excludes data loading, H2D and multi-GPU communication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import numpy as np
import torch
from hydra import compose, initialize_config_dir


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=["off", "manual"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--catalog-snapshot",
        action="store_true",
        help="Read the existing catalog as an explicit diagnostic snapshot, even if its config fingerprint is stale",
    )
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", default="phase1a_distogram_v110",
                        help="Hydra config name under configs/miniworld (e.g. phase1a_distogram_v120)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra Hydra overrides, e.g. data.train_db.catalog_cache_path=...")
    args = parser.parse_args()
    from run_miniworld_distogram_train import Config
    from miniworld.data.dataloader.dataloader import (
        BioMolData,
        _load_catalog_arrow,
        _catalog_fingerprint,
    )
    from miniworld.models.distogram_only import MiniSWAModel
    from miniworld.loss.auxiliary import cal_atom_distogram_loss
    from miniworld.training.engine_backend import configure_engine_backend
    from miniworld.utils import get_step_decay_scheduler_with_warmup
    from cudagraph_trainer import _load_static
    from miniworld_engine.autotune.native import source_identity

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        cfg = Config.model_validate(
            compose(
                config_name=args.config,
                overrides=["train.engine_backend=triton", *args.override],
            )
        )
    log(f"config={args.config} bucket_msa={cfg.train.bucket_msa_multiple} "
        f"msa_subsample_per_recycle={cfg.model.trunk.msa_subsample_per_recycle} "
        f"pool_by_source={dict(cfg.data.msa.max_msa_depth_by_source)} "
        f"policy_by_source={dict(cfg.data.msa.sample_depth_by_source)}")
    configure_engine_backend("triton")
    if not args.batch.exists():
        log("Loading real training batches from the existing catalog (read only)")

        class ReadOnlyCatalogData(BioMolData):
            def _load_items(self):
                path = Path(self.config.DB_config.catalog_cache_path)
                items, weights, sources, raw, fingerprint = _load_catalog_arrow(path)
                if not raw:
                    raise RuntimeError("Catalog has legacy balanced weights")
                if fingerprint != _catalog_fingerprint(
                    self.config.DB_config, self.config.sampler_config
                ):
                    if not args.catalog_snapshot:
                        raise RuntimeError(
                            "Catalog mismatch: pass --catalog-snapshot for a read-only diagnostic snapshot"
                        )
                    from miniworld.data.dataloader.dataloader import (
                        source_balanced_weights_from_sources,
                        configured_source_weights,
                    )

                    log(
                        f"Explicit diagnostic catalog snapshot: {path}; fingerprint={fingerprint}"
                    )
                    self.items = items
                    self.weights = source_balanced_weights_from_sources(
                        sources=sources,
                        raw_weights=weights,
                        source_weights=configured_source_weights(self.config.DB_config),
                        default_source_weight=self.config.DB_config.default_source_weight,
                    )
                else:
                    super()._load_items()

        data = ReadOnlyCatalogData(
            BioMolData.BioMolConfig(
                crop_config=cfg.data.crop,
                msa_config=cfg.data.msa,
                DB_config=cfg.data.train_db,
                sampler_config=cfg.data.sampler,
                tokenizer_config=cfg.data.tokenizer,
            )
        )
        loader = data.create_ddp_dataloader(
            world_size=1,
            rank=0,
            seed=17,
            drop_last=False,
            batch_size=1,
            num_workers=0,
            num_samples_per_rank=2,
            shuffle=True,
            bucket_msa_multiple=cfg.train.bucket_msa_multiple,
            bucket_token_multiple=cfg.train.bucket_token_multiple,
            bucket_atom_multiple=cfg.train.bucket_atom_multiple,
            bucket_template_multiple=4,
        )
        data.set_epoch(0)
        batches = list(loader)
        torch.save(batches, args.batch)
        del data, loader
    else:
        batches = torch.load(args.batch, weights_only=False, map_location="cpu")
    log(f"Loaded {len(batches)} cached real batches")
    batch_hash = hashlib.sha256(args.batch.read_bytes()).hexdigest()
    source_batches = [b.to(device="cuda") for b in batches]
    # Separate static buffers: every arm includes the identical GPU-to-GPU batch copies.
    import copy

    static = copy.deepcopy(source_batches[0])
    torch.manual_seed(17)
    model = MiniSWAModel(cfg.model).cuda().train()
    model._forced_n_recycle = args.recycles
    initial_hash = hashlib.sha256()
    for p in model.parameters():
        initial_hash.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    backend_counts = {}
    for m in model.modules():
        if hasattr(m, "_backend"):
            backend_counts[m._backend.value] = (
                backend_counts.get(m._backend.value, 0) + 1
            )
    assert set(backend_counts) == {"triton"}, backend_counts
    opt = torch.optim.Adam(model.parameters(), cfg.train.max_lr, betas=(0.9, 0.95))
    sched = get_step_decay_scheduler_with_warmup(
        optimizer=opt,
        warmup_steps=cfg.train.warmup_steps,
        decay_steps=cfg.train.decay_steps,
        decay_factor=cfg.train.decay_factor,
    )
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model_state = model.state_dict()
    log(
        f"Model params={sum(p.numel() for p in model.parameters())} backends={backend_counts}; recycle={args.recycles}; graph={args.graph}"
    )

    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 512
        model.compile(dynamic=False, options={"triton.cudagraphs": False})

    def micro():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                msa=static.msa,
                reference=static.reference,
                scheme=static.scheme,
                sequence=static.sequence,
                structure=static.structure,
                template=static.template,
            )
            loss = cfg.loss.distogram_loss * cal_atom_distogram_loss(
                logits,
                static.structure.atom_pos,
                static.structure.atom_pos_mask,
                static.scheme.atom_to_token_idx_map,
                rep_atom_mask=static.structure.atom_is_rep,
                token_asym_id=static.scheme.token_asym_id,
                interchain_weight=cfg.loss.distogram_interchain_weight,
            )
        loss.backward()
        return loss

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start_prepare = time.perf_counter()
    with torch.cuda.stream(stream):
        for i in range(3):
            opt.zero_grad(set_to_none=False)
            loss = micro()
            torch.cuda.synchronize()
            log(
                f"warmup {i + 1}/3 loss={loss.item():.6f} peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            )
        opt.zero_grad(set_to_none=False)
        graph = None
        capture_seconds = 0.0
        if args.graph == "manual":
            capture_start = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_loss = micro()
            torch.cuda.synchronize()
            capture_seconds = time.perf_counter() - capture_start
            log(f"Capture complete in {capture_seconds:.3f}s")
        torch.cuda.synchronize()
        prepare_seconds = time.perf_counter() - start_prepare
        opt.zero_grad(set_to_none=False)
        torch.manual_seed(2026)
        batch_index = 0

        def train_step():
            nonlocal batch_index
            losses = []
            for _ in range(args.grad_accum):
                _load_static(static, source_batches[batch_index % len(source_batches)])
                batch_index += 1
                if graph is None:
                    value = micro()
                else:
                    graph.replay()
                    value = graph_loss
                losses.append(value.detach().clone())
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip_max_norm
            )
            opt.step()
            sched.step()
            with torch.no_grad():
                for k, v in ema.items():
                    v.mul_(cfg.train.ema_decay).add_(
                        model_state[k], alpha=1 - cfg.train.ema_decay
                    )
            opt.zero_grad(set_to_none=False)
            return torch.stack(losses).mean(), norm

        log("Full optimizer-step warmup")
        warm_loss, warm_norm = train_step()
        torch.cuda.synchronize()
        assert torch.isfinite(warm_loss) and torch.isfinite(warm_norm)
        log(f"Step warmup loss={warm_loss.item():.6f} grad_norm={warm_norm.item():.6f}")
        preparation_peak_allocated = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rounds = []
        for i in range(args.rounds):
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            begin.record(stream)
            loss, norm = train_step()
            end.record(stream)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            assert torch.isfinite(loss) and torch.isfinite(norm)
            record = dict(
                round=i,
                wall_seconds=seconds,
                cuda_seconds=begin.elapsed_time(end) / 1000,
                loss=loss.item(),
                grad_norm=norm.item(),
            )
            rounds.append(record)
            log(json.dumps(record))
        import gc
        from triton.runtime.autotuner import Autotuner
        choices = sorted(set(
            (o.base_fn.__module__ + "." + o.base_fn.__name__, str(key), str(config))
            for o in gc.get_objects() if type(o) is Autotuner
            for key, config in o.cache.items()
        ))
        result = dict(
            graph=args.graph,
            recycles=args.recycles,
            compile=args.compile,
            engine_backend="triton",
            device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            engine_identity=source_identity(),
            initial_weights_sha256=initial_hash.hexdigest(),
            batch_sha256=batch_hash,
            batch_shapes={
                name: {
                    k: list(v.shape)
                    for k, v in vars(getattr(static, name)).items()
                    if isinstance(v, torch.Tensor)
                }
                for name in ["msa", "structure", "scheme", "template"]
            },
            config_name=args.config,
            config=cfg.model_dump(mode="json"),
            parameter_count=sum(p.numel() for p in model.parameters()),
            backend_counts=backend_counts,
            grad_accum=args.grad_accum,
            preparation_seconds=prepare_seconds,
            capture_seconds=capture_seconds,
            autotune_choices=choices,
            autotune_choices_sha256=hashlib.sha256(json.dumps(choices).encode()).hexdigest(),
            peak_allocated_including_preparation_gib=max(preparation_peak_allocated, torch.cuda.max_memory_allocated()) / 2**30,
            device_used_gib=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            warmup_loss=warm_loss.item(),
            warmup_grad_norm=warm_norm.item(),
            rounds=rounds,
            median_step_seconds=statistics.median(r["wall_seconds"] for r in rounds),
            scope="full Phase1 model forward/loss/backward + GPU batch copy + grad accumulation + Adam + clip + scheduler + EMA; excludes data loading/H2D/DDP; fixed recycle; two real batches repeatedly used; random initialized weights",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        log(f"Saved {args.output}")
    torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    main()
