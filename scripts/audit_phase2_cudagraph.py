"""Measure a real phase-2 frozen trunk, with CUDA graph launch evidence.

Run each --graph arm in a fresh process on the same GPU. Only cudagraphs differ
between off/trees. Compilation, autotuning and capture are outside timing. Output
clones are included in every arm, as required by the downstream training consumer.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.setrecursionlimit(30000)


def load_model_batch(config, checkpoint, weights="model", catalog_snapshot=None):
    from run_miniworld_diffusion_train import Config
    from miniworld.data.dataloader.dataloader import BioMolData
    from miniworld.models.diffusion.model import DiffusionModel

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    config = Path(config).resolve()
    with initialize_config_dir(str(config.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=config.name))
    model = DiffusionModel(cfg.model).cuda().train()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    sd = dict(ck["model_state_dict"])
    if weights == "ema":
        sd.update(ck["ema_state_dict"])
    model.load_state_dict(sd, strict=True)
    model.requires_grad_(False)
    print(
        f"Loaded {weights} weights: epoch={ck.get('epoch')} step={ck.get('global_step')}",
        flush=True,
    )
    dataset_type = BioMolData
    if catalog_snapshot is not None:
        # An explicitly selected, read-only diagnostic snapshot. Its raw sampling
        # weights are used as stored; this is not the training catalog rebuild path.
        from miniworld.data.dataloader.dataloader import _load_catalog_arrow

        class SnapshotData(BioMolData):
            def _load_items(self):
                self.items, self.weights, _, _, fingerprint = _load_catalog_arrow(
                    Path(catalog_snapshot)
                )
                print(
                    f"Diagnostic catalog snapshot: {catalog_snapshot} fingerprint={fingerprint}",
                    flush=True,
                )

        dataset_type = SnapshotData
    data = dataset_type(
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
        seed=0,
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
    batch = next(iter(loader)).to(device="cuda")
    return model, batch, cfg, ck


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "relative_l2": (delta.norm() / b.float().norm().clamp_min(1e-20)).item(),
        "finite": bool(torch.isfinite(a).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=("model", "ema"), default="model")
    ap.add_argument("--graph", choices=("off", "trees", "manual"), required=True)
    ap.add_argument("--recycles", type=int, default=2)
    ap.add_argument("--rep", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force-split", action="store_true")
    ap.add_argument("--catalog-snapshot", type=Path)
    ap.add_argument("--fullgraph", action="store_true")
    ap.add_argument(
        "--paired",
        action="store_true",
        help="Compare OFF/ON on identical tensors in this process",
    )
    args = ap.parse_args()
    from miniworld_engine import settings

    # The engine no longer reads MINIWORLD_TRANSITION_FORCE_SPLIT.
    settings.configure(transition_force_split=args.force_split)
    print(f"ENGINE SETTINGS {settings.current()}", flush=True)
    model, batch, cfg, ck = load_model_batch(
        args.config, args.ckpt, args.weights, args.catalog_snapshot
    )
    model._forced_n_recycle = args.recycles
    fn_args = (
        batch.msa,
        batch.reference,
        batch.scheme,
        batch.sequence,
        batch.structure,
        batch.template,
    )
    metadata = {
        "graph": args.graph,
        "recycles": args.recycles,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "weights": args.weights,
        "checkpoint": args.ckpt,
        "epoch": ck.get("epoch"),
        "amp": "bf16-mixed",
        "tokens": int(batch.token_length),
        "atoms": int(batch.atom_length),
        "msa": int(batch.msa_depth),
        "force_split": args.force_split,
        "fullgraph": args.fullgraph,
    }
    del ck
    print(f"METADATA {json.dumps(metadata)}", flush=True)
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 512
    counters = Counter()
    replay, capture = torch.cuda.CUDAGraph.replay, torch.cuda.CUDAGraph.capture_begin

    def replay_count(graph):
        counters["replay"] += 1
        return replay(graph)

    def capture_count(graph, *a, **kw):
        counters["capture"] += 1
        return capture(graph, *a, **kw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        print("Eager preflight/autotune", flush=True)
        eager = tuple(t.clone() for t in model._condition_impl(*fn_args))
        torch.cuda.synchronize()
        fn = torch.compile(
            model._condition_impl,
            dynamic=False,
            fullgraph=args.fullgraph,
            options={"triton.cudagraphs": args.graph == "trees"},
        )

        def invoke():
            torch.compiler.cudagraph_mark_step_begin()
            return tuple(t.clone() for t in fn(*fn_args))

        with (
            patch.object(torch.cuda.CUDAGraph, "replay", replay_count),
            patch.object(torch.cuda.CUDAGraph, "capture_begin", capture_count),
        ):
            print("Compile and warm up", flush=True)
            stable = 0
            for i in range(12):
                before = counters.copy()
                t0 = time.perf_counter()
                result = invoke()
                torch.cuda.synchronize()
                changes = counters - before
                print(
                    f"warm {i}: {time.perf_counter() - t0:.3f}s {dict(changes)}",
                    flush=True,
                )
                del result
                stable = stable + 1 if not changes["capture"] else 0
                if i >= 3 and stable >= 3:
                    break
            reference = invoke()
            metadata["vs_eager"] = [difference(a, b) for a, b in zip(reference, eager)]
            # Keep the compiled non-graph reference for same-kernel parity across arms.
            if args.graph == "off":
                torch.save(
                    tuple(t.cpu() for t in reference), args.output.with_suffix(".pt")
                )
            del eager, reference
            if args.graph == "manual":
                graph = torch.cuda.CUDAGraph()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        static_out = fn(*fn_args)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    static_out = fn(*fn_args)

                def invoke():
                    graph.replay()
                    return tuple(t.clone() for t in static_out)

                for _ in range(3):
                    invoke()
            torch.cuda.synchronize()
            times = []
            gpu_times = []
            before = counters.copy()
            for _ in range(args.rep):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                t0 = time.perf_counter()
                start.record()
                result = invoke()
                end.record()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                gpu_times.append(start.elapsed_time(end) / 1000)
                del result
            metadata["measurement_counts"] = dict(counters - before)
            metadata["wall_seconds"] = times
            metadata["gpu_seconds"] = gpu_times
            metadata["median_seconds"] = statistics.median(times)
            metadata["median_gpu_seconds"] = statistics.median(gpu_times)
            metadata["warmup_counts"] = dict(before)
            print(f"TIMING {json.dumps(metadata)}", flush=True)
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(3):
                    invoke()
                torch.cuda.synchronize()
            trace_path = args.output.with_suffix(".trace.json")
            prof.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
            runtime = Counter(
                e.get("name")
                for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
            )
            metadata["profile_runtime"] = dict(runtime)
            metadata["profile_cudaGraphLaunch"] = sum(
                v for k, v in runtime.items() if "GraphLaunch" in k
            )
            metadata["graph_breaks"] = dict(torch._dynamo.utils.counters["graph_break"])
            metadata["unimplemented"] = dict(
                torch._dynamo.utils.counters["unimplemented"]
            )
            if args.paired and args.graph != "off":
                print("Paired same-input comparison and changed-input check", flush=True)
                fn_off = torch.compile(
                    model._condition_impl,
                    dynamic=False,
                    fullgraph=args.fullgraph,
                    options={"triton.cudagraphs": False},
                )

                def invoke_off():
                    torch.compiler.cudagraph_mark_step_begin()
                    return tuple(t.clone() for t in fn_off(*fn_args))

                for _ in range(4):
                    invoke_off()
                    invoke()
                a, b = invoke_off(), invoke()
                metadata["paired_output_difference"] = [
                    difference(x, y) for x, y in zip(b, a)
                ]
                metadata["paired_output_equal"] = all(
                    torch.equal(x, y) for x, y in zip(a, b)
                )
                again_off, again_on = invoke_off(), invoke()
                metadata["repeat_off_difference"] = [
                    difference(x, y) for x, y in zip(a, again_off)
                ]
                metadata["repeat_on_difference"] = [
                    difference(x, y) for x, y in zip(b, again_on)
                ]
                del again_off, again_on
                del a, b
                paired = {"off": [], "on": []}
                before = counters.copy()
                for i in range(args.rep):
                    order = (("off", invoke_off), ("on", invoke))
                    for label, call in order if i % 2 == 0 else order[::-1]:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        result = call()
                        torch.cuda.synchronize()
                        paired[label].append(time.perf_counter() - t0)
                        del result
                metadata["paired_seconds"] = paired
                metadata["paired_counts"] = dict(counters - before)
                metadata["paired_speedup"] = statistics.median(
                    paired["off"]
                ) / statistics.median(paired["on"])
                original = batch.reference.pos.clone()
                baseline = invoke()
                batch.reference.pos[..., 0].mul_(1.1)
                changed_off, changed_on = invoke_off(), invoke()
                metadata["changed_input_difference"] = [
                    difference(x, y) for x, y in zip(changed_on, changed_off)
                ]
                metadata["changed_input_equal"] = all(
                    torch.equal(x, y) for x, y in zip(changed_on, changed_off)
                )
                metadata["changed_input_has_effect"] = any(
                    not torch.equal(x, y) for x, y in zip(baseline, changed_on)
                )
                repeated_off, repeated_on = invoke_off(), invoke()
                metadata["changed_input_repeat_off"] = [
                    difference(x, y) for x, y in zip(repeated_off, changed_off)
                ]
                metadata["changed_input_repeat_on"] = [
                    difference(x, y) for x, y in zip(repeated_on, changed_on)
                ]
                valid = batch.structure.token_mask.bool()
                valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
                metadata["changed_input_valid_pair_difference"] = difference(
                    changed_on[1][valid_pair], changed_off[1][valid_pair]
                )
                batch.reference.pos.copy_(original)
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"RESULT {json.dumps(metadata)}", flush=True)


if __name__ == "__main__":
    main()
