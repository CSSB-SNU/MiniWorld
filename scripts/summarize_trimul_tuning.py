"""Summarize observed TriMul cache keys and compare the native build driver."""

import argparse, json, importlib, collections
import torch
from pathlib import Path
from miniworld_engine.autotune import cache, cache_status
from miniworld_engine.autotune.configs import configs_for

parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
base = args.output_dir
base.mkdir(parents=True, exist_ok=True)
files = sorted(args.inputs)
logs = [json.loads(p.read_text()) for p in files]
observed = {}
for d in logs:
    for r in d["lookups"]:
        if r["kind"] not in ("triton", "native"):
            continue
        key = (r["kind"], r["op"], r["dtype"], r["bucket"])
        observed.setdefault(key, dict(record=r, cases=[]))["cases"].append(
            dict(
                family=d["family"],
                length=r["length"],
                training=r["training"],
                compiled=r["compiled"],
            )
        )
report = []
for (kind, op, dtype, bucket), item in observed.items():
    r = dict(
        kind=kind,
        op=op,
        dtype=dtype,
        bucket=bucket,
        cache_hit=item["record"]["cache_hit"],
        cases=item["cases"],
    )
    data = cache._load(op, logs[0]["gpu"])
    r["file_exists"] = data is not None
    if kind == "triton" and data:
        module, symbol = cache_status._registry_symbols()[op]
        tuner = getattr(importlib.import_module(module), symbol)
        identity = cache.implementation_identity(tuner)
        grid = {repr(cache._sig(c)) for c in configs_for(op)}
        profiles = data.get("measurements", {}).get(f"{dtype}|{bucket}", {})
        searched = set()
        matching = []
        for p in profiles.values():
            if p["workload"].get("implementation") not in (None, identity):
                continue
            searched.update(p.get("searched", []))
            matching.append(p["workload"].get("arguments", {}))
        r.update(
            grid_size=len(grid),
            searched_in_current_grid=len(grid & searched),
            unsearched_current_grid=len(grid - searched),
            matching_profiles=len(matching),
            provenance=data["provenance"],
            representative_arguments=matching,
        )
    else:
        r.update(
            fallback=item["record"].get("fallback"),
            candidate_count=item["record"].get("candidate_count"),
        )
    report.append(r)
(base / "runtime_cache_coverage.json").write_text(json.dumps(report, indent=2))
print(
    "Unique runtime keys:",
    collections.Counter((r["kind"], r["cache_hit"]) for r in report),
)
print("Triton ops:", len(set(r["op"] for r in report if r["kind"] == "triton")))
print("Unsearched entries:", sum(bool(r.get("unsearched_current_grid")) for r in report))
for op in sorted(set(r["op"] for r in report)):
    rs = [r for r in report if r["op"] == op]
    print(op, len(rs), "unsearched", [r.get("unsearched_current_grid") for r in rs])

# Meta tensors execute only the driver shape construction, with no GPU launches.
import json, importlib, ast
from pathlib import Path
from miniworld_engine.autotune.native import tensor_key
from miniworld_engine.kernels.drivers import trimul_inproj as driver

module = importlib.import_module(
    "miniworld_engine.kernels.trimul_inproj.cute.masked_front"
)
captured = []


def record(a, b, mask, save):
    normalized_mask = mask.reshape(1, a.shape[0]).to(torch.float32).contiguous()
    captured.append(tensor_key(a, b, normalized_mask, extra=(save,)))


module.masked_front = record
driver.dev = lambda: "meta"
driver.D = 128
for length in (128, 384, 768):
    driver.L = length
    driver.M = length * length
    driver.masked_front_sm90()
rows = report
result = []
for r in rows:
    if r["kind"] != "native":
        continue
    result.append(
        dict(
            bucket=r["bucket"], driver_can_emit=r["bucket"] in captured, cases=r["cases"]
        )
    )
(base / "native_driver_coverage.json").write_text(
    json.dumps(dict(driver_keys=captured, runtime=result), indent=2)
)
print(
    "Driver can emit",
    sum(r["driver_can_emit"] for r in result),
    "of",
    len(result),
    "observed native runtime keys",
)
for r in result:
    c = r["cases"][0]
    print(
        c["family"],
        c["length"],
        "train" if c["training"] else "infer",
        r["driver_can_emit"],
    )
