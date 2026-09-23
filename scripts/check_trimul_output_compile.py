"""Check that builder CPU precompile contracts match the new runtime ABI."""

from pathlib import Path
import json
import torch
from miniworld_engine.autotune import native, native_compile
from miniworld_engine.autotune.cute_config import plain_sm90_candidates, config_to_kwargs

m, k, n = 264, 256, 128
tensors = [
    torch.empty(m, n, dtype=torch.bfloat16),
    torch.empty(n, k, dtype=torch.bfloat16),
    torch.empty(m, k, dtype=torch.bfloat16),
    torch.empty(k, dtype=torch.bfloat16),
    *[torch.empty(m, dtype=torch.float32) for _ in range(3)],
]
bucket = native.tensor_key(*tensors)
op = "trimul_output_bwd_rows_sm90_cute"
space = plain_sm90_candidates()
assert len(native.candidates_for(op, bucket)) == len(space) == 512
tasks = [
    native_compile.task_for(op, config_to_kwargs(space[i]), bucket)
    for i in (0, 2, 64, 128, 256, 384)
]
result = native_compile.run_tasks(
    tasks,
    jobs=3,
    directory="runs/trimul_output_opt_20260916/compile_contract",
    timeout=300,
)
Path("runs/trimul_output_opt_20260916/compile_contract.json").write_text(
    json.dumps(result, indent=2)
)
assert all(row["status"] == "ok" for row in result.values()), result
print("CPU PRECOMPILE CONTRACTS PASSED", len(result))
