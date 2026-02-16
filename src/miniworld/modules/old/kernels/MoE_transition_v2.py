import torch
import triton
import triton.language as tl
import os

from einops import rearrange

from team_gm.utils.MoE_utils import (
    loss_free_route,
    group_by_expert,
    scatter_expert,
    stack_expert,
)
from team_gm.modules.old.kernels.utils import (
    get_seq_group,
    STANDARD_CONFIGS,
    early_config_prune,
    _compute_pid,
)
from team_gm.modules.old.kernels.layernorm import (
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
    layer_norm_fwd_fused_recal,
)
from team_gm.modules.old.kernels.MoE_GEMM_meta import (
    cg_grouped_gemm_backward_weights,
    cg_grouped_gemm_backward_inputs,
)


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "transition"

if AUTOTUNE:
    configs = []
    for BM in [16, 32, 64]:
        for BK in [16, 32, 64]:
            if BM * BK > 2048:
                continue
            for s in [1, 2, 3]:
                for w in [4, 8, 16]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_K": BK},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
else:
    configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    ]


# fmt: off
@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def MoE_expand_fwd_kernel(
    x_ptr, w1_ptr, w2_ptr, indices_ptr, out_ptr,
    M, N: tl.constexpr, K: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    '''
    x : (M,K)
    w : (N,K)
    out : (M,N)
    '''
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_M
        n_start = tile_n_idx * BLOCK_N

        # Only process if in bounds
        if m_start < M:
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_n = n_start + tl.arange(0, BLOCK_N)

            acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # Determine the expert group index and load expert ID
            group_idx = m_start // GROUP_SIZE_M
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

            for ki in range(k_tiles):
                # Offsets for K dim
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

                # Create masks for bounds checking
                mask_m = offs_m < M
                mask_n = offs_n < N
                mask_k = offs_k < K

                # masks for A and B
                mask_x = mask_m[:, None] & mask_k[None, :]
                mask_w = mask_n[:, None] & mask_k[None, :]

                # Load inputs (A) with bounds checking
                x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
                x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                w1_ptrs = (
                    w1_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )

                w2_ptrs = (
                    w2_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                w1, w2 = tl.load(w1_ptrs, mask=mask_w, other=0.0), tl.load(w2_ptrs, mask=mask_w, other=0.0)

                # Accumulate matrix multiplication for this K tile
                acc1 += tl.dot(x, w1.T)
                acc2 += tl.dot(x, w2.T)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Create masks for bounds checking
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask_out= mask_m[:, None] & mask_n[None, :]

            out = acc1 * tl.sigmoid(acc1) * acc2

            # Store output (C) with bounds checking
            out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptrs, out, mask=mask_out)

# fmt: on


# fmt: off
@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def MoE_squeeze_fwd_kernel(
    x_ptr, score_ptr, w_ptr, indices_ptr, out_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    '''
    x : (M,K)
    w : (N,K)
    out : (M,N)
    '''
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_M
        n_start = tile_n_idx * BLOCK_N

        # Only process if in bounds
        if m_start < M:
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_n = n_start + tl.arange(0, BLOCK_N)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # Determine the expert group index and load expert ID
            group_idx = m_start // GROUP_SIZE_M
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

            score_tile = tl.load(score_ptr + offs_m, mask=(offs_m < M), other=0.0)

            for ki in range(k_tiles):
                # Offsets for K dim
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

                # Create masks for bounds checking
                mask_m = offs_m < M
                mask_n = offs_n < N
                mask_k = offs_k < K

                # masks for A and B
                mask_x = mask_m[:, None] & mask_k[None, :]
                mask_w = mask_n[:, None] & mask_k[None, :]

                # Load inputs (A) with bounds checking
                x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
                x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                w_ptrs = (
                    w_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                w = tl.load(w_ptrs, mask=mask_w, other=0.0)


                # Accumulate matrix multiplication for this K tile
                acc += tl.dot(x, w.T)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Create masks for bounds checking
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask_out= mask_m[:, None] & mask_n[None, :]

            out = acc * score_tile[:, None]

            # Store output (C) with bounds checking
            out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptrs, out, mask=mask_out)

# fmt: on


# fmt: off
@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def MoE_expand_bwd_kernel(
    x_ptr, grad_out_ptr, indices_ptr,
    w1_ptr, w2_ptr,
    dA_ptr, dB_ptr,
    M, N: tl.constexpr, K: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    '''
    x : (M,K)
    w1,w2,w3 : (N,K)
    '''
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_M
        n_start = tile_n_idx * BLOCK_N

        # Only process if in bounds
        if m_start < M:
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_n = n_start + tl.arange(0, BLOCK_N)

            grad_out_ptrs = grad_out_ptr + (offs_m[:, None] * N + offs_n[None, :])
            grad_out = tl.load(grad_out_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)  # (BLOCK_M, BLOCK_N)

            # Determine the expert group index and load expert ID
            group_idx = m_start // GROUP_SIZE_M
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)
            mask_m = offs_m < M
            mask_n = offs_n < N

            A_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            B_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(k_tiles):
                # Offsets for K dim
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

                # Create masks for bounds checking
                mask_k = offs_k < K

                # masks for A and B
                mask_x = mask_m[:, None] & mask_k[None, :]
                mask_w = mask_n[:, None] & mask_k[None, :]

                # Load inputs (A) with bounds checking
                x_bias = offs_m[:, None] * K + offs_k[None, :]
                x_ptrs = x_ptr + x_bias
                x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                w_bias = expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                w1_ptrs = w1_ptr + w_bias
                w2_ptrs = w2_ptr + w_bias
                w1 = tl.load(w1_ptrs, mask=mask_w, other=0.0)
                w2 = tl.load(w2_ptrs, mask=mask_w, other=0.0)

                # Accumulate matrix multiplication for this K tile
                A_acc += tl.dot(x, w1.T, allow_tf32=False)
                B_acc += tl.dot(x, w2.T, allow_tf32=False)

            sigmoid_A = tl.sigmoid(A_acc)
            swish_A = A_acc * sigmoid_A
            swish_diff_A = sigmoid_A + swish_A * (1 - sigmoid_A)  # (BLOCK_M, n*d)
            dA_tile = grad_out * B_acc * swish_diff_A
            dB_tile = grad_out * A_acc * sigmoid_A  # (BLOCK_M, n*d)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Create masks for bounds checking
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask_out= mask_m[:, None] & mask_n[None, :]

            out_bias = (offs_m[:, None] * N + offs_n[None, :])
            dA_ptrs = dA_ptr + out_bias
            dB_ptrs = dB_ptr + out_bias
            tl.store(dA_ptrs,dA_tile,mask=mask_out)
            tl.store(dB_ptrs,dB_tile,mask=mask_out)

# fmt: on
# fmt: off
@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def MoE_squeeze_bwd_kernel(
    grad_out_ptr, score_ptr, w_ptr, indices_ptr, out_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    '''
    grad_out : (M,K)
    w : (N,K)
    out : (M,N)
    '''
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_M
        n_start = tile_n_idx * BLOCK_N

        # Only process if in bounds
        if m_start < M:
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_n = n_start + tl.arange(0, BLOCK_N)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # Determine the expert group index and load expert ID
            group_idx = m_start // GROUP_SIZE_M
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

            score_tile = tl.load(score_ptr + offs_m, mask=(offs_m < M), other=0.0)

            for ki in range(k_tiles):
                # Offsets for K dim
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

                # Create masks for bounds checking
                mask_m = offs_m < M
                mask_n = offs_n < N
                mask_k = offs_k < K

                # masks for A and B
                mask_x = mask_m[:, None] & mask_k[None, :]
                mask_w = mask_n[:, None] & mask_k[None, :]

                # Load inputs (A) with bounds checking
                grad_out_ptrs = grad_out_ptr + offs_m[:, None] * K + offs_k[None, :]
                x = tl.load(grad_out_ptrs, mask=mask_x, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                w_ptrs = (
                    w_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                w = tl.load(w_ptrs, mask=mask_w, other=0.0)


                # Accumulate matrix multiplication for this K tile
                acc += tl.dot(x, w.T)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Create masks for bounds checking
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask_out= mask_m[:, None] & mask_n[None, :]

            out = acc * score_tile[:, None]

            # Store output (C) with bounds checking
            out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptrs, out, mask=mask_out)

class TritonMoETransitionExpandFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sorted_y: torch.Tensor,
        expand_a_weight: torch.Tensor,  # (E, N, n*N)
        expand_b_weight: torch.Tensor,  # (E, N, n*N)
        expert_map: torch.Tensor,  # (M,)
        n: int,
        k: int,
    ):
        op_dtype = sorted_y.dtype
        N = sorted_y.shape[1]
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        # expand = torch.empty(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)
        expand = torch.zeros(
            sorted_y.shape[0], n * N, dtype=op_dtype, device=sorted_y.device
        )

        # fmt: off
        M = sorted_y.shape[0]

        # Calculate grid size for the kernel
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid = (NUM_SMS, 1, 1)
        MoE_expand_fwd_kernel[grid](
            sorted_y, _expand_a_weight, _expand_b_weight, expert_map, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        ctx.save_for_backward(
            sorted_y.bfloat16(),
            _expand_a_weight,
            _expand_b_weight,
            expert_map,
        )
        ctx.n = n
        ctx.num_experts = _expand_a_weight.shape[0]
        ctx.op_dtype = op_dtype

        return expand

    @staticmethod
    def backward(ctx, grad_expand: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            sorted_y,
            expand_a_weight,
            expand_b_weight,
            expert_map,
        ) = ctx.saved_tensors
        sorted_y = sorted_y.to(op_dtype)
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        n = ctx.n
        num_experts = ctx.num_experts
        M, N = sorted_y.shape

        grad_expand = rearrange(grad_expand, "... d -> (...) d").contiguous()

        # grad_H
        dA = torch.zeros_like(grad_expand)
        dB = torch.zeros_like(grad_expand)

        # fmt: off
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        MoE_expand_bwd_kernel[grid](
            sorted_y, grad_expand, expert_map,
            _expand_a_weight, _expand_b_weight,
            dA, dB,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on
        dW_a = cg_grouped_gemm_backward_weights(
            dA, sorted_y, expert_map, num_experts=num_experts, group_size_m=128
        )
        dW_b = cg_grouped_gemm_backward_weights(
            dB, sorted_y, expert_map, num_experts=num_experts, group_size_m=128
        )

        dy_expert_a = torch.zeros_like(sorted_y)
        dy_expert_b = torch.zeros_like(sorted_y)

        dy_expert_a = cg_grouped_gemm_backward_inputs(dA, _expand_a_weight, expert_map)
        dy_expert_b = cg_grouped_gemm_backward_inputs(dB, _expand_b_weight, expert_map)

        dy_expert = dy_expert_a + dy_expert_b
        dx = dy_expert

        return (
            dx,
            dW_a.float(),
            dW_b.float(),
            None,
            None,
            None,
        )


class TritonMoETransitionSqueezeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        expand: torch.Tensor,
        score: torch.Tensor,
        squeeze_weight: torch.Tensor,  # (E, N, n*N)
        expert_map: torch.Tensor,  # (M,)
        n: int,
        k: int,
    ):
        op_dtype = expand.dtype
        M = expand.shape[0]
        N = expand.shape[1] // n
        _squeeze_weight = squeeze_weight.to(op_dtype)
        squeeze = torch.zeros(M, N, dtype=op_dtype, device=expand.device)

        # fmt: off

        # Calculate grid size for the kernel
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        MoE_squeeze_fwd_kernel[grid](
            expand, score, _squeeze_weight, expert_map, squeeze,
            M, N, n*N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        ctx.save_for_backward(
            expand.bfloat16(),
            score.bfloat16(),
            squeeze_weight,
            expert_map,
        )
        ctx.n = n
        ctx.num_experts = squeeze_weight.shape[0]
        ctx.op_dtype = op_dtype

        return expand

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            expand,
            score,
            squeeze_weight,
            expert_map,
            expert_map,
        ) = ctx.saved_tensors
        expand = expand.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)
        n = ctx.n
        num_experts = ctx.num_experts
        M, K = expand.shape
        N = K // n

        grad_out = rearrange(grad_out, "... d -> (...) d").contiguous()

        grad_expand = torch.zeros(M, n * N, dtype=op_dtype, device=grad_out.device)
        _squeeze_weight_t = _squeeze_weight.transpose(-1,-2).contiguous()

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        MoE_squeeze_bwd_kernel[grid](
            grad_out, score, _squeeze_weight_t, expert_map, grad_expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        dW_s = cg_grouped_gemm_backward_weights(
            grad_out, expand, expert_map, num_experts=num_experts, group_size_m=128
        )

        return (
            grad_expand,
            None,
            dW_s.float(),
            None,
            None,
            None,
        )


triton_MoE_transition_expand = TritonMoETransitionExpandFunction.apply
