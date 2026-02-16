import torch
import triton
import triton.language as tl
import os

from einops import rearrange

from miniworld.utils.MoE_utils import (
    loss_free_route,
    group_by_expert,
    scatter_expert,
    stack_expert,
)
from miniworld.modules.old.kernels.utils import (
    get_seq_group,
    early_config_prune,
    _compute_pid,
)
from miniworld.modules.old.kernels.layernorm import (
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
    layer_norm_fwd_fused_recal,
)
from miniworld.modules.old.kernels.MoE_GEMM_meta import (
    cg_grouped_gemm_forward,
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
                            {"BLOCK_M": BM, "BLOCK_N": BM, "BLOCK_K": BK},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
else:
    configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    ]

# fmt: off
@triton.autotune(
    configs=configs,
    key=["GROUP_M", "N"],
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
    GROUP_M: tl.constexpr,
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
    configs=configs,
    key=["GROUP_M", "N"],
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
    GROUP_M: tl.constexpr,
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
    configs=configs,
    key=["GROUP_M", "N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def MoE_bwd_kernel(
    x_ptr, grad_out_ptr, indices_ptr,
    w1_ptr, w2_ptr, w3_ptr,
    dA_ptr, dB_ptr, expand_ptr,
    M, N: tl.constexpr, K: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
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

            dH_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Determine the expert group index and load expert ID
            group_idx = m_start // GROUP_SIZE_M
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)
            mask_m = offs_m < M
            mask_n = offs_n < N

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
                grad_out_ptrs = grad_out_ptr + x_bias
                grad_out = tl.load(grad_out_ptrs, mask=mask_x, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                w_bias = expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                w3_ptrs = w3_ptr + w_bias
                w3 = tl.load(w3_ptrs, mask=mask_w, other=0.0)

                # Accumulate matrix multiplication for this K tile
                dH_acc += tl.dot(grad_out, w3.T, allow_tf32=False)

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
            swish_AB = swish_A * B_acc
            dA_tile = dH_acc * B_acc * swish_diff_A
            dB_tile = A_acc * sigmoid_A * dH_acc  # (BLOCK_M, n*d)

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
            expand_ptrs = expand_ptr + out_bias
            dA_ptrs = dA_ptr + out_bias
            dB_ptrs = dB_ptr + out_bias
            tl.store(expand_ptrs, swish_AB, mask=mask_out)
            tl.store(dA_ptrs,dA_tile,mask=mask_out)
            tl.store(dB_ptrs,dB_tile,mask=mask_out)

# fmt: on


class TritonMoETransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        router_weight: torch.Tensor,  # (E, N)
        expand_a_weight: torch.Tensor,  # (E, N, n*N)
        expand_b_weight: torch.Tensor,  # (E, N, n*N)
        squeeze_weight: torch.Tensor,  # (E, n*N, N)
        expert_frequency: torch.Tensor,
        n: int,
        k: int,
    ):
        orig_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        ln_weight = ln_weight.to(op_dtype)
        ln_bias = ln_bias.to(op_dtype)
        _router_weight = router_weight.to(op_dtype)
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)

        M, N = x.shape

        y = torch.empty_like(x)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x, y, ln_weight, ln_bias, mean, rstd,
            x.stride(0), x.stride(1),
            M, N, 1e-5,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on
        topk_score, topk_indices = loss_free_route(
            expert_frequency, _router_weight, y, k
        )  # (M, k)

        # sort y
        padding = 128  # Warning : padding should be larger than block m
        sorted_y, sorted_score, idx_map, expert_map, _, _ = group_by_expert(
            y, topk_score, topk_indices, padding=padding
        )

        # expand = torch.empty(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)
        expand = torch.zeros(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)

        # fmt: off
        M = sorted_y.shape[0]

        # Calculate grid size for the kernel
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid = (NUM_SMS, 1, 1)
        MoE_expand_fwd_kernel[grid](
            sorted_y, _expand_a_weight, _expand_b_weight, expert_map, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        # fmt: off
        squeeze = torch.zeros(sorted_y.shape[0], N, dtype=op_dtype, device=x.device)
        MoE_squeeze_fwd_kernel[grid](
            expand, sorted_score, _squeeze_weight, expert_map, squeeze,
            M, N, n*N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        output = scatter_expert(x.shape, squeeze, idx_map)
        output = output.reshape(orig_shape)

        ctx.save_for_backward(
            x.bfloat16(),
            ln_weight,
            ln_bias,
            router_weight,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            mean,
            rstd,
            topk_score,
            topk_indices,
        )
        ctx.n = n
        ctx.k = k
        ctx.num_experts = _expand_a_weight.shape[0]
        ctx.op_dtype = op_dtype
        ctx.orig_shape = orig_shape

        return output, topk_indices

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_indices: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            x,
            ln_weight,
            ln_bias,
            router_weight,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            mean,
            rstd,
            topk_score,
            topk_indices,
        ) = ctx.saved_tensors
        x = x.to(op_dtype)
        grad_output = grad_output.to(op_dtype)
        _router_weight = router_weight.to(op_dtype)
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)
        topk_score = topk_score.to(op_dtype)
        n = ctx.n
        k = ctx.k
        num_experts = ctx.num_experts
        M, N = x.shape

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()

        y = torch.empty_like(x)

        # fmt: off
        layernorm_grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused_recal[layernorm_grid](
            x, y, ln_weight, ln_bias, mean, rstd,
            x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        # sort y
        padding = 128
        sorted_y, sorted_score, idx_map, expert_map, pos, m_sel = group_by_expert(
            y, topk_score, topk_indices, padding=padding
        )
        sorted_grad = torch.zeros_like(sorted_y)
        valid = m_sel >= 0
        sorted_grad[pos[valid]] = grad_output[m_sel[valid]]

        M = sorted_y.shape[0]
        sorted_gs = sorted_grad * sorted_score[:, None]

        # grad_H
        expand = torch.zeros(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)
        dA = torch.zeros_like(expand)
        dB = torch.zeros_like(expand)

        # fmt: off
        _squeeze_weight_t = _squeeze_weight.clone().transpose(-1,-2).contiguous()
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        MoE_bwd_kernel[grid](
            sorted_y, sorted_gs, expert_map,
            _expand_a_weight, _expand_b_weight, _squeeze_weight_t,
            dA, dB, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on
        dW_s = cg_grouped_gemm_backward_weights(
            sorted_gs,
            expand,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )

        dW_a = cg_grouped_gemm_backward_weights(
            dA,
            sorted_y,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )
        dW_b = cg_grouped_gemm_backward_weights(
            dB,
            sorted_y,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )

        dy_expert_a = torch.zeros_like(sorted_y)
        dy_expert_b = torch.zeros_like(sorted_y)

        dy_expert_a = cg_grouped_gemm_backward_inputs(dA, _expand_a_weight, expert_map)
        dy_expert_b = cg_grouped_gemm_backward_inputs(dB, _expand_b_weight, expert_map)

        dy_expert = dy_expert_a + dy_expert_b

        dy_expert = scatter_expert(x.shape, dy_expert, idx_map)

        # gradient from router

        # fmt: off
        grid = (NUM_SMS, 1, 1)
        squeeze = torch.zeros(sorted_y.shape[0], N, dtype=op_dtype, device=x.device)
        MoE_squeeze_fwd_kernel[grid](
            expand, sorted_score, _squeeze_weight, expert_map, squeeze,
            M, N, n*N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        gs = torch.einsum("bd,bd->b", sorted_grad, squeeze)

        _gs_sum = scatter_expert((x.shape[0],), gs, idx_map)
        gs_sum = torch.zeros_like(gs)
        gs_sum[pos[valid]] = _gs_sum[m_sel[valid]]
        dr = gs - sorted_score * gs_sum
        dr = stack_expert((x.shape[0], num_experts), dr, idx_map, expert_map)  # (L, E)

        dW_r = torch.matmul(dr.T, y)  # (E, N)
        dy_router = torch.matmul(dr, _router_weight.contiguous())  # (L, N)

        dy = dy_expert + dy_router

        dy = dy.reshape(orig_shape)
        dw = torch.zeros(N, dtype=torch.float32, device=dy.device)
        db = torch.zeros(N, dtype=torch.float32, device=dy.device)
        dx = torch.empty_like(dy)

        # fmt: off
        M = dx.shape[0]
        layernorm_grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_bwd_dx_fused[layernorm_grid](
            dx, dy, dw, db,
            x, ln_weight, mean, rstd,
            dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        return (
            dx,
            dw.float(),
            db.float(),
            dW_r.float(),
            dW_a.float(),
            dW_b.float(),
            dW_s.float(),
            None,
            None,
            None,
        )


class TritonMoETransitionWoLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        router_weight: torch.Tensor,  # (E, N)
        expand_a_weight: torch.Tensor,  # (E, N, n*N)
        expand_b_weight: torch.Tensor,  # (E, N, n*N)
        squeeze_weight: torch.Tensor,  # (E, n*N, N)
        expert_frequency: torch.Tensor,  # (E,)
        n: int,
        k: int,
    ):
        orig_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        _router_weight = router_weight.to(op_dtype)
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)

        M, N = x.shape

        topk_score, topk_indices = loss_free_route(
            expert_frequency, _router_weight, x, k
        )  # (M, k)

        # sort y
        padding = 128  # Warning : padding should be larger than block m
        sorted_y, sorted_score, idx_map, expert_map, _, _ = group_by_expert(
            x, topk_score, topk_indices, padding=padding
        )

        # expand = torch.empty(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)
        expand = torch.zeros(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)

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

        # fmt: off
        squeeze = torch.zeros(sorted_y.shape[0], N, dtype=op_dtype, device=x.device)
        MoE_squeeze_fwd_kernel[grid](
            expand, sorted_score, _squeeze_weight, expert_map, squeeze,
            M, N, n*N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        output = scatter_expert(x.shape, squeeze, idx_map)
        output = output.reshape(orig_shape)

        ctx.save_for_backward(
            x.bfloat16(),
            router_weight,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            topk_score,
            topk_indices,
        )
        ctx.n = n
        ctx.k = k
        ctx.num_experts = _expand_a_weight.shape[0]
        ctx.op_dtype = op_dtype
        ctx.orig_shape = orig_shape

        return output, topk_indices

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_indices: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            x,
            router_weight,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            topk_score,
            topk_indices,
        ) = ctx.saved_tensors
        x = x.to(op_dtype)
        grad_output = grad_output.to(op_dtype)
        _router_weight = router_weight.to(op_dtype)
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)
        topk_score = topk_score.to(op_dtype)
        n = ctx.n
        k = ctx.k
        num_experts = ctx.num_experts
        M, N = x.shape

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()

        # sort y
        padding = 128
        sorted_y, sorted_score, idx_map, expert_map, pos, m_sel = group_by_expert(
            x, topk_score, topk_indices, padding=padding
        )
        sorted_grad = torch.zeros_like(sorted_y)
        valid = m_sel >= 0
        sorted_grad[pos[valid]] = grad_output[m_sel[valid]]
        M = sorted_y.shape[0]
        sorted_gs = sorted_grad * sorted_score[:, None]

        # grad_H
        expand = torch.zeros(sorted_y.shape[0], n * N, dtype=op_dtype, device=x.device)
        dA = torch.zeros_like(expand)
        dB = torch.zeros_like(expand)

        # fmt: off
        _squeeze_weight_t = _squeeze_weight.clone().transpose(-1,-2).contiguous()
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        MoE_bwd_kernel[grid](
            sorted_y, sorted_gs, expert_map,
            _expand_a_weight, _expand_b_weight, _squeeze_weight_t,
            dA, dB, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on
        dW_s = cg_grouped_gemm_backward_weights(
            sorted_gs,
            expand,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )

        dW_a = cg_grouped_gemm_backward_weights(
            dA,
            sorted_y,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )
        dW_b = cg_grouped_gemm_backward_weights(
            dB,
            sorted_y,
            expert_map,
            num_experts=num_experts,
            group_size_m=128,
        )

        dy_expert_a = torch.zeros_like(sorted_y)
        dy_expert_b = torch.zeros_like(sorted_y)

        dy_expert_a = cg_grouped_gemm_backward_inputs(dA, _expand_a_weight, expert_map)
        dy_expert_b = cg_grouped_gemm_backward_inputs(dB, _expand_b_weight, expert_map)

        dy_expert = dy_expert_a + dy_expert_b

        dy_expert = scatter_expert(x.shape, dy_expert, idx_map)

        # gradient from router

        # recaluclate squeeze

        grid = (NUM_SMS, 1, 1)
        squeeze = torch.zeros(sorted_y.shape[0], N, dtype=op_dtype, device=x.device)
        MoE_squeeze_fwd_kernel[grid](
            expand,
            sorted_score,
            _squeeze_weight,
            expert_map,
            squeeze,
            M,
            N,
            n * N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        gs = torch.einsum("bd,bd->b", sorted_grad, squeeze)

        _gs_sum = scatter_expert((x.shape[0],), gs, idx_map)
        gs_sum = torch.zeros_like(gs)
        gs_sum[pos[valid]] = _gs_sum[m_sel[valid]]
        dr = gs - sorted_score * gs_sum
        dr = stack_expert((x.shape[0], num_experts), dr, idx_map, expert_map)  # (L, E)

        dW_r = torch.matmul(dr.T, x)  # (E, N)
        dy_router = torch.matmul(dr, _router_weight.contiguous())  # (L, N)

        dy = dy_expert + dy_router
        dy = dy.reshape(orig_shape)

        return (
            dy,
            dW_r.float(),
            dW_a.float(),
            dW_b.float(),
            dW_s.float(),
            None,
            None,
            None,
        )


triton_MoE_transition = TritonMoETransitionFunction.apply
triton_MoE_transition_wo_ln = TritonMoETransitionWoLNFunction.apply
