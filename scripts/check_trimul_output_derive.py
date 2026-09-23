"""Exercise the real module planner's FakeTensor path, including fast-path selection."""

import json
from pathlib import Path
from miniworld_engine.autotune import derive
from miniworld_engine.autotune.builder import cases

derive.require_environment()
derive.install_no_calibration()
derive.install_module_apply()
derive.install_native_recorders()
from miniworld_engine.kernels.trimul_inproj.cute import output_training

calls = []
original = output_training.forward


def traced(x, *args):
    calls.append(x.shape[0])
    return original(x, *args)


output_training.forward = traced
catalog = {c.name: c for c in cases()}
result = []
for length in (128, 384, 768):
    unit = derive.DeriveUnit(
        "triangle_multiplication_bidirectional",
        "token_pair",
        length,
        (("d_pair", 128), ("d_hidden", 128)),
        "miniworld",
        "bfloat16",
        "",
        "train",
        None,
    )
    launches, error = derive.record(unit, catalog)
    result.append({"length": length, "error": error, "launches": launches})
Path("runs/trimul_output_opt_20260916/derive.json").write_text(
    json.dumps({"units": result, "native_rows": calls}, indent=2)
)
assert all(r["error"] is None for r in result), result
assert calls == [384**2, 768**2], calls
print("PLANNER AND SHAPE POLICY PASSED")
