import triton


def get_seq_group(L):
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
    for length in GROUP_LENGTHS:
        if L <= length:
            return length
    return GROUP_LENGTHS[-1]


STANDARD_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    # Medium sizes
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    # Larger sizes with more warps, stages
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_stages=4,
        num_warps=8,
    ),
]


def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    k = kwargs.get("K", 0)
    valid_configs = [
        config for config in configs if config.kwargs.get("BLOCK_K", 0) <= k
    ]
    # If all configs were filtered out, return at least one config
    if not valid_configs and configs:
        # Find the config with the smallest BLOCK_K
        return [min(configs, key=lambda c: c.kwargs.get("BLOCK_K", float("inf")))]

    return valid_configs


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n
