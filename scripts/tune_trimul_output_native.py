"""Measure the declared CuTe output GEMM spaces on one GPU, with resumable evidence."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import torch
import triton
from miniworld_engine.autotune.cute_config import (
    fused_lnl_candidates,
    plain_sm90_candidates,
)
from miniworld_engine.autotune import native, capture, cache
from miniworld_engine.autotune.cute_config import config_to_kwargs
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import dgrad_ln_rows

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--groups", default="backward")
p.add_argument("--record-shard", type=Path)
p.add_argument("--runtime-trace", type=Path)
args = p.parse_args()
M = args.length**2
K, N = 256, 128
torch.manual_seed(424)
torch.backends.cuda.matmul.allow_tf32 = False
x = torch.randn(K, M, device="cuda", dtype=torch.bfloat16).t()
g = (torch.randn(K, device="cuda") * 0.15 + 1).bfloat16().float()
b = (torch.randn(K, device="cuda") * 0.1).bfloat16()
w = (torch.randn(N, K, device="cuda") * 0.03).bfloat16()
dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
mean = x.float().mean(1)
rs = torch.rsqrt(x.float().var(1, unbiased=False) + 1e-5)
xhat = ((x.float() - mean[:, None]) * rs[:, None]).bfloat16().contiguous()
refy = (
    torch.nn.functional.layer_norm(x.float(), (K,), g.float(), b.float(), 1e-5)
    @ w.float().t()
)
dxn = dy.float() @ w.float()
dxhat = dxn * g.float()
c1 = (dxhat * xhat.float()).mean(1)
c2 = dxhat.mean(1)
refdx = rs[:, None] * (dxhat - c2[:, None] - xhat.float() * c1[:, None])
result = (
    json.loads(args.output.read_text())
    if args.output.exists()
    else {
        "L": args.length,
        "gpu": torch.cuda.get_device_name(),
        "forward": [],
        "backward": [],
    }
)
identity = native.source_identity()
if args.runtime_trace:
    observed = json.loads(args.runtime_trace.read_text())
    buckets = {
        row["bucket"]
        for row in observed["cache"]
        if row["op"] == "trimul_output_bwd_rows_sm90_cute" and "bucket" in row
    }
    assert buckets == {native.tensor_key(dy, w, xhat, g, rs, c1, c2)}, (
        "tuning operands differ from the actual compiled training ABI",
        buckets,
    )
    result["runtime_abi_verified"] = str(args.runtime_trace)
if args.record_shard:
    assert not result["backward"] or result.get("source_identity") == identity, (
        "do not relabel earlier source measurements"
    )
    result["source_identity"] = identity
    bucket = native.tensor_key(dy, w, xhat, g, rs, c1, c2)
    assert not result["backward"] or result.get("bucket") == bucket, (
        "do not reuse timings for a different operand ABI"
    )
    result["bucket"] = bucket
    capture.reset()


def save():
    args.output.write_text(json.dumps(result, indent=2))


def error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8))


for group, configs in [
    ("forward", fused_lnl_candidates()),
    ("backward", plain_sm90_candidates()),
]:
    if group not in args.groups.split(","):
        continue
    done = {r["index"] for r in result[group]}
    for i, cfg in enumerate(configs):
        if i in done:
            continue
        begin = time.time()
        row = {"index": i, "config": dataclasses.asdict(cfg)}
        try:
            if group == "forward":
                fn = lambda: layernorm_linear_cute_fused(
                    x, g, b, w, None, return_stats=True, config=cfg
                )
                out, mu, rstd = fn()
                errors = [error(out, refy), error(mu, mean), error(rstd, rs)]
            else:
                fn = lambda: dgrad_ln_rows(dy, w, xhat, g, rs, c1, c2, config=cfg)
                out = fn()
                errors = [error(out, refdx)]
            assert max(errors) < 0.008, errors
            ms = triton.testing.do_bench_cudagraph(fn, rep=15)
            row.update(ms=ms, rel_l2=errors)
        except Exception as e:
            row["error"] = str(e)
            print("REJECT", group, i, type(e).__name__, str(e)[:200], flush=True)
            if (
                "illegal memory" in str(e).lower()
                or "misaligned address" in str(e).lower()
            ):
                result[group].append(row)
                save()
                raise
        row["seconds"] = time.time() - begin
        result[group].append(row)
        save()
        if i % 8 == 0:
            valid = [r for r in result[group] if "ms" in r]
            print(
                group,
                i + 1,
                "/",
                len(configs),
                "best",
                min(valid, key=lambda r: r["ms"])["ms"] if valid else None,
                flush=True,
            )
    result[group + "_best"] = min(
        (r for r in result[group] if "ms" in r), key=lambda r: r["ms"]
    )
    save()
    print("BEST", group, result[group + "_best"], flush=True)
if args.record_shard:
    assert args.groups == "backward"
    assert all("ms" in r for r in result["backward"]), (
        "failed candidates cannot be published as complete"
    )
    assert len(result["backward"]) == len(plain_sm90_candidates())
    measurement = {
        "scheme": 1,
        "kind": "native",
        "implementation": identity,
        "timing": "cuda_graph",
        "bench_clear_mb": 0,
        "bench_rep_ms": 15,
        "harness": "trimul-output-native-v1",
    }
    bucket = native.tensor_key(dy, w, xhat, g, rs, c1, c2)
    grid = [{"kwargs": config_to_kwargs(c)} for c in plain_sm90_candidates()]
    for row in result["backward"]:
        capture.record_native(
            "trimul_output_bwd_rows_sm90_cute",
            grid,
            "bfloat16",
            bucket,
            config_to_kwargs(plain_sm90_candidates()[row["index"]]),
            row["ms"],
            identity,
            measurement=measurement,
        )
    capture.dump_shard(str(args.record_shard), unit_complete=True)
    merged = capture.merge_shards(
        [str(args.record_shard)],
        gpu=cache.gpu_key(),
        only_ops={"trimul_output_bwd_rows_sm90_cute"},
    )
    assert merged and not capture._MERGE_SKIPPED, (merged, capture._MERGE_SKIPPED)
    result["published"] = merged
    save()
