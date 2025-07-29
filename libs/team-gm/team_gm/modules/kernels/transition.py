import torch
import triton
import triton.language as tl
import os

from einops import rearrange

from .utils import get_seq_group
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
@triton.autotune(configs=configs, key=["GROUP_M", "n", "N"])
@triton.jit
def transition_fwd_kernel(
    x_ptr, W1_ptr, W2_ptr, out_ptr,
    M, n: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    offs_n = tl.arange(0, n * N)  # full row indices

    # Accumulators in float32.
    A_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)

    # Tiled GEMM over the k dimension.
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )

        A_tile += tl.dot(x_tile, W1_tile, allow_tf32=False)
        B_tile += tl.dot(x_tile, W2_tile, allow_tf32=False)

    swish_A = A_tile * tl.sigmoid(A_tile)
    swish_AB = swish_A * B_tile

    out_ptr_ = out_ptr + (offs_m[:, None] * n * N + offs_n[None, :])
    tl.store(out_ptr_, swish_AB, mask=(offs_m[:, None] < M))
# fmt: on


# fmt: off
@triton.autotune(configs=configs, key=["GROUP_M", "n", "N"])
@triton.jit
def transition_bwd_kernel(
    x_ptr, W1_ptr, W2_ptr, grad_expand_ptr, dA_ptr, dB_ptr,
    M, n: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    offs_n = tl.arange(0, n * N)  # full row indices

    # Accumulators in float32.
    A_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)

    # Tiled GEMM over the k dimension.
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )

        A_tile += tl.dot(x_tile, W1_tile, allow_tf32=False)
        B_tile += tl.dot(x_tile, W2_tile, allow_tf32=False)

    grad_expand_tile = tl.load(
        grad_expand_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        mask=(offs_m[:, None] < M),
        other=0.0,
    )

    sigmoid_A = tl.sigmoid(A_tile)
    swish_diff_A = sigmoid_A + A_tile * sigmoid_A * (1 - sigmoid_A)  # (BLOCK_M, n*d)
    dA_tile = grad_expand_tile * B_tile * swish_diff_A
    dB_tile = A_tile * sigmoid_A * grad_expand_tile  # (BLOCK_M, n*d)

    tl.store(
        dA_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dA_tile,
        mask=(offs_m[:, None] < M),
    )

    tl.store(
        dB_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dB_tile,
        mask=(offs_m[:, None] < M),
    )
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
        x = rearrange(x, "... d -> (...) d").contiguous()
        M, N = x.shape
        if N != 128:
            raise ValueError(
                f"Triton transition design to support only 128 dimension, but got {N}"
            )

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

        expand = torch.empty(M, n * N, dtype=x.dtype, device=x.device)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        transition_fwd_kernel[grid](
            y, expand_a_weight, expand_b_weight, expand,
            M, n, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            ln_weight,
            ln_bias,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            mean,
            rstd,
        )
        ctx.n = n

        output = torch.matmul(expand, squeeze_weight.T)
        output = output.reshape(orig_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
        x = x.to(grad_output.dtype)
        n = ctx.n
        M, N = x.shape

        x_out = torch.empty_like(x)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused_recal[grid](
            x, x_out, ln_weight, ln_bias, mean, rstd,
            x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        expand = torch.empty(M, n * N, dtype=x_out.dtype, device=x_out.device)

        # fmt: off
        transition_fwd_kernel[grid](
            x_out, expand_a_weight, expand_b_weight, expand,
            M, n, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()
        grad_expand = torch.matmul(grad_output, squeeze_weight)
        grad_squeeze_weight = torch.matmul(grad_output.T, expand)
        dA = torch.empty(M, n * N, dtype=x_out.dtype, device=x_out.device)
        dB = torch.empty(M, n * N, dtype=x_out.dtype, device=x_out.device)

        # fmt: off
        transition_bwd_kernel[grid](
            x_out, expand_a_weight, expand_b_weight, grad_expand, dA, dB,
            M, n, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        grad_a_weight = torch.matmul(dA.T, x_out)
        grad_b_weight = torch.matmul(dB.T, x_out)
        dy = torch.matmul(dA, expand_a_weight) + torch.matmul(dB, expand_b_weight)
        dy = dy.reshape(orig_shape)
        dw = torch.zeros(N, dtype=torch.float32, device=dy.device)
        db = torch.zeros(N, dtype=torch.float32, device=dy.device)
        dx = torch.empty_like(dy)

        # fmt: off
        layer_norm_bwd_dx_fused[grid](
            dx, dy, dw, db,
            x, ln_weight, mean, rstd,
            dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        return dx, dw, db, grad_a_weight, grad_b_weight, grad_squeeze_weight, None


triton_transition = TritonTransitionFunction.apply
