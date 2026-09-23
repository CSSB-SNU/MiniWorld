from miniworld_engine.autotune.cute_config import gated_sm90_candidates, config_to_kwargs
from miniworld_engine.autotune.native_compile import task_for, run_tasks

config = config_to_kwargs(gated_sm90_candidates()[0])
tasks = []
for save in (False, True):
    tensors = (
        ((192, 128), (128, 1), "torch.bfloat16"),
        ((128, 512), (512, 1), "torch.bfloat16"),
        ((192,), (1,), "torch.bool"),
    )
    tasks.append(
        task_for("trimul_inproj_masked_sm90_cute", config, repr((tensors, (save,))))
    )
from quack.gemm_config import GemmConfig

config = config_to_kwargs(
    GemmConfig(
        tile_m=128,
        tile_n=256,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
        device_capacity=9,
    )
)
tensors = (
    ((16384, 2048), (2048, 1), "torch.bfloat16"),
    ((512, 2048), (2048, 1), "torch.bfloat16"),
    ((16384, 512), (512, 1), "torch.bfloat16"),
)
tasks.append(
    task_for("transition_squeeze_residual_sm90_cute", config, repr((tensors, ())))
)
results = run_tasks(
    tasks, jobs=2, directory="runs/h100_upgrade/native_contract", timeout=300
)
assert all(r["status"] == "ok" for r in results.values()), results
