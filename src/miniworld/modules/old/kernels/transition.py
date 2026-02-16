import torch
import triton
import triton.language as tl
import os

from einops import rearrange
from miniworld.modules.old.kernels.utils import (
    get_seq_group,
    STANDARD_CONFIGS,
    early_config_prune,
    _compute_pid,
)
from .layernorm import (
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused,
    layer_norm_fwd_fused_recal,
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
    key=["GROUP_M", "N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def transition_fwd_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    M, N: tl.constexpr, K: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
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
                    w1_ptr + offs_n[:, None] * K + offs_k[None, :]
                )

                w2_ptrs = (
                    w2_ptr + offs_n[:, None] * K + offs_k[None, :]
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
    key=["GROUP_M", "N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def transition_bwd_kernel(
    x_ptr, grad_expand_ptr,
    w1_ptr, w2_ptr,
    dA_ptr, dB_ptr, expand_ptr,
    M, N: tl.constexpr, K: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
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

            dH_acc = tl.load(
                grad_expand_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M),
                other=0.0,
            )

            # Determine the expert group index and load expert ID
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
                w_bias = offs_n[:, None] * K + offs_k[None, :]
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
            expand = swish_A * B_acc

            dA_tile = dH_acc * B_acc * swish_diff_A
            dB_tile = dH_acc *  A_acc * sigmoid_A  # (BLOCK_M, n*d)

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
            tl.store(expand_ptrs, expand, mask=mask_out)
            tl.store(dA_ptrs,dA_tile,mask=mask_out)
            tl.store(dB_ptrs,dB_tile,mask=mask_out)
# fmt: on


class TritonTransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        expand_a_weight: torch.Tensor,
        expand_b_weight: torch.Tensor,
        squeeze_weight: torch.Tensor,
        n: int,
    ):
        orig_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        ln_weight = ln_weight.to(op_dtype)
        ln_bias = ln_bias.to(op_dtype)
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

        expand = torch.empty(M, n * N, dtype=op_dtype, device=x.device)

        # fmt: off
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        transition_fwd_kernel[grid](
            y, _expand_a_weight, _expand_b_weight, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        ctx.save_for_backward(
            x.bfloat16(),
            ln_weight,
            ln_bias,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            mean,
            rstd,
        )
        ctx.n = n
        ctx.op_dtype = op_dtype

        output = torch.matmul(expand, _squeeze_weight.T)
        output = output.reshape(orig_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            x,
            ln_weight,
            ln_bias,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            mean,
            rstd,
        ) = ctx.saved_tensors
        x = x.to(op_dtype)
        expand_a_weight = expand_a_weight.to(op_dtype)
        expand_b_weight = expand_b_weight.to(op_dtype)
        squeeze_weight = squeeze_weight.to(op_dtype)
        n = ctx.n
        M, N = x.shape

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

        expand = torch.empty(M, n * N, dtype=op_dtype, device=y.device)

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()
        grad_expand = torch.matmul(grad_output, squeeze_weight)
        dA = torch.empty(M, n * N, dtype=y.dtype, device=y.device)
        dB = torch.empty(M, n * N, dtype=y.dtype, device=y.device)

        # fmt: off
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        transition_bwd_kernel[grid](
            y,
            grad_expand,
            expand_a_weight, expand_b_weight,
            dA, dB, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        grad_a_weight = torch.matmul(dA.T, y)
        grad_b_weight = torch.matmul(dB.T, y)
        dy = torch.matmul(dA, expand_a_weight) + torch.matmul(dB, expand_b_weight)
        dy = dy.reshape(orig_shape)
        dw = torch.zeros(N, dtype=torch.float32, device=dy.device)
        db = torch.zeros(N, dtype=torch.float32, device=dy.device)
        dx = torch.empty_like(dy)

        # fmt: off
        layer_norm_bwd_dx_fused[layernorm_grid](
            dx, dy, dw, db,
            x, ln_weight, mean, rstd,
            dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on
        grad_squeeze_weight = torch.matmul(grad_output.T, expand)

        return (
            dx,
            dw.float(),
            db.float(),
            grad_a_weight.float(),
            grad_b_weight.float(),
            grad_squeeze_weight.float(),
            None,
        )


class TritonTransitionWoLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expand_a_weight: torch.Tensor,
        expand_b_weight: torch.Tensor,
        squeeze_weight: torch.Tensor,
        n: int,
    ):
        orig_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        _expand_a_weight = expand_a_weight.to(op_dtype)
        _expand_b_weight = expand_b_weight.to(op_dtype)
        _squeeze_weight = squeeze_weight.to(op_dtype)

        M, N = x.shape

        expand = torch.empty(M, n * N, dtype=op_dtype, device=x.device)

        # fmt: off
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        transition_fwd_kernel[grid](
            x, _expand_a_weight, _expand_b_weight, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        ctx.save_for_backward(
            x.bfloat16(),
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        )
        ctx.n = n
        ctx.op_dtype = op_dtype

        output = torch.matmul(expand, _squeeze_weight.T)
        output = output.reshape(orig_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        op_dtype = ctx.op_dtype
        (
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        ) = ctx.saved_tensors
        x = x.to(op_dtype)
        expand_a_weight = expand_a_weight.to(op_dtype)
        expand_b_weight = expand_b_weight.to(op_dtype)
        squeeze_weight = squeeze_weight.to(op_dtype)
        n = ctx.n
        M, N = x.shape

        expand = torch.empty(M, n * N, dtype=op_dtype, device=x.device)

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()
        squeeze_weight_t = squeeze_weight.T.contiguous()
        dA = torch.empty(M, n * N, dtype=x.dtype, device=x.device)
        dB = torch.empty(M, n * N, dtype=x.dtype, device=x.device)

        # fmt: off
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (NUM_SMS, 1, 1)
        transition_bwd_kernel[grid](
            x, grad_output,
            expand_a_weight, expand_b_weight, squeeze_weight_t,
            dA, dB, expand,
            M, n*N, N,
            NUM_SMS=NUM_SMS,
        )
        # fmt: on

        grad_a_weight = torch.matmul(dA.T, x)
        grad_b_weight = torch.matmul(dB.T, x)
        dx = torch.matmul(dA, expand_a_weight) + torch.matmul(dB, expand_b_weight)
        dx = dx.reshape(orig_shape)
        # fmt: on
        grad_squeeze_weight = torch.matmul(grad_output.T, expand)

        return (
            dx,
            grad_a_weight.float(),
            grad_b_weight.float(),
            grad_squeeze_weight.float(),
            None,
        )


triton_transition = TritonTransitionFunction.apply
triton_transition_wo_ln = TritonTransitionWoLNFunction.apply
