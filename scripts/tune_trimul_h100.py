"""Measure all declared masked-front SM90 candidates, validate math, emit a shard."""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture, native
from miniworld_engine.autotune.cute_config import config_to_kwargs, gated_sm90_candidates

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--cuda-graph", action="store_true")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
settings.configure(run_autotune=True, compile_jobs=8)
torch.backends.cuda.matmul.allow_tf32 = False
module = importlib.import_module(
    "miniworld_engine.kernels.trimul_inproj.cute.masked_front"
)
original_launch = module._launch
validated = {}
current = {}
results = []


def launch_checked(a, b, out, preact, mask, config):
    original_launch(a, b, out, preact, mask, config)
    key = json.dumps(config_to_kwargs(config), sort_keys=True)
    if key not in validated:
        reference = current["reference"]
        expected = current["expected"]
        output_error = float(
            (out.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        )
        preact_error = None
        if preact is not None:
            preact_error = float(
                (preact.float() - reference).norm() / reference.norm().clamp_min(1e-8)
            )
        if not output_error < 0.025 or (
            preact_error is not None and not preact_error < 0.025
        ):
            raise RuntimeError(
                f"Candidate numerical failure: {output_error}, {preact_error}"
            )
        assert (out[~current["valid_mask"]] == 0).all(), "masked rows not zero"
        validated[key] = dict(
            output_relative_l2=output_error, preact_relative_l2=preact_error
        )


module._launch = launch_checked


def tune_graph(a, b, mask, save):
    """All candidates share operands; alternate order and rank median graph time."""
    op = "trimul_inproj_masked_sm90_cute"
    normalized_mask = mask.reshape(1, a.shape[0]).float().contiguous()
    bucket = native.tensor_key(a, b, normalized_mask, extra=(save,))
    configs = gated_sm90_candidates()
    grid = [dict(kwargs=config_to_kwargs(c)) for c in configs]
    out, preact = module._masked_front_fake(a, b, mask, save)
    graphs = []
    for config in configs:
        launch_checked(a, b, out, preact if save else None, normalized_mask, config)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            original_launch(a, b, out, preact if save else None, normalized_mask, config)
        for _ in range(3):
            graph.replay()
        graphs.append(graph)
    samples = [[] for _ in graphs]
    for repeat in range(12):
        order = list(range(len(graphs)))
        if repeat % 2:
            order.reverse()
        for i in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(50):
                graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) / 50)
    for config, values in zip(configs, samples):
        kw = config_to_kwargs(config)
        capture.record_native(
            op,
            grid,
            "bfloat16",
            bucket,
            kw,
            statistics.median(values),
            native.source_identity(),
        )
        validated[json.dumps(kw, sort_keys=True)]["graph_samples_ms"] = values
    return out, preact


for projected, save, label in [
    (256, False, "inference"),
    (512, True, "single_training"),
    (1024, True, "bidir_training"),
]:
    torch.manual_seed(891)
    m = args.length**2
    a = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, projected, device="cuda", dtype=a.dtype) * 0.08
    mask = torch.ones(m, device="cuda", dtype=torch.bool)
    mask[::3] = False
    ref = a.float() @ b.float()
    expected = torch.sigmoid(ref[:, ::2]) * ref[:, 1::2] * mask[:, None]
    current.update(reference=ref, expected=expected, valid_mask=mask)
    validated.clear()
    started = time.monotonic()
    out, preact = (
        tune_graph(a, b, mask, save)
        if args.cuda_graph
        else module.masked_front(a, b, mask, save)
    )
    torch.cuda.synchronize()
    results.append(
        dict(
            length=args.length,
            projected=projected,
            save_preact=save,
            label=label,
            measurement="cuda_graph" if args.cuda_graph else "cold_cache_events",
            seconds=time.monotonic() - started,
            validated_configs=dict(validated),
        )
    )
    print(label, "validated configs", len(validated), flush=True)
    capture.dump_shard(str(args.output_dir / "measurements.json"))
    (args.output_dir / "validation.json").write_text(json.dumps(results, indent=2))
    del out, preact, a, b, mask, ref, expected
    current.clear()
    torch.cuda.empty_cache()
capture.dump_shard(str(args.output_dir / "measurements.json"), unit_complete=True)
print("Complete:", args.length, native.source_identity(), flush=True)
