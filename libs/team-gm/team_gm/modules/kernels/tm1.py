import torch
import triton
import triton.language as tl
import os

from einops import rearrange

from .utils import get_seq_group


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"

if AUTOTUNE:
    fwd_configs = []
    for BM in [16, 32, 64]:
        for BK in [16, 32, 64]:
            if BM * BK > 2048:
                continue
            for s in [2, 3]:
                for w in [4, 8]:
                    fwd_configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_K": BK},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
    bwd_configs = []
    for BM in [16, 32, 64, 128]:
        for BK in [16, 32, 64, 128]:
            if BM * BK > 2048:
                continue
            for s in [2, 3]:
                for w in [4, 8]:
                    bwd_configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_K": BK},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
else:
    fwd_configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ]


# fmt: off
@triton.autotune(configs=fwd_configs, key=["GROUP_M", "N"])
@triton.jit
def fused_sigmoid_gate_fwd_kernel(
    x_ptr, WLg_ptr, WL_ptr, WRg_ptr, WR_ptr,
    left_ptr, right_ptr,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    offs_d_out = tl.arange(0, N)  # full column indices for output

    # Accumulators in float32.
    LA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    # Tiled GEMM over the k dimension.
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load x tile: (BLOCK_M, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        # Load W1 tile: (BLOCK_K, d)
        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        # Compute partial A and B using dot product (accumulates in float32)
        LA_tile += tl.dot(x_tile, WLg_tile, allow_tf32=False)
        LB_tile += tl.dot(x_tile, WL_tile, allow_tf32=False)
        RA_tile += tl.dot(x_tile, WRg_tile, allow_tf32=False)
        RB_tile += tl.dot(x_tile, WR_tile, allow_tf32=False)

    # Compute sigmoid and output (element-wise in float32)
    Lg_tile = tl.sigmoid(LA_tile)
    left_tile = Lg_tile * LB_tile
    Rg_tile = tl.sigmoid(RA_tile)
    right_tile = Rg_tile * RB_tile

    # Store the output tiles
    tl.store(
        left_ptr + (offs_m[:, None] * N + offs_d_out[None, :]),
        left_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        right_ptr + (offs_m[:, None] * N + offs_d_out[None, :]),
        right_tile,
        mask=(offs_m[:, None] < M),
    )
# fmt: on


# fmt: off
@triton.autotune(configs=bwd_configs, key=["GROUP_M", "d"])
@triton.jit
def fused_sigmoid_gate_bwd_kernel(
    x_ptr, WLg_ptr, WL_ptr, WRg_ptr, WR_ptr,
    grad_left_out_ptr, grad_right_out_ptr, dLA_ptr, dLB_ptr, dRA_ptr, dRB_ptr,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # --- Setup ---
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    offs_d_full = tl.arange(0, N)  # full dimension indices [d]

    # --- Recompute A and B in float32 (Memory Saving Strategy) ---
    # (This part remains the same as previous optimized version)
    LA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        LA_tile += tl.dot(x_tile, WLg_tile, allow_tf32=False)
        LB_tile += tl.dot(x_tile, WL_tile, allow_tf32=False)
        RA_tile += tl.dot(x_tile, WRg_tile, allow_tf32=False)
        RB_tile += tl.dot(x_tile, WR_tile, allow_tf32=False)

    # --- Compute dA and dB (Element-wise in float32) ---
    # (This part remains the same as previous optimized version)
    Lg_tile = tl.sigmoid(LA_tile)
    Rg_tile = tl.sigmoid(RA_tile)
    grad_left_tile = tl.load(
        grad_left_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=(offs_m[:, None] < M),
    )
    grad_right_tile = tl.load(
        grad_right_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=(offs_m[:, None] < M),
    )
    dLB_tile = grad_left_tile * Lg_tile
    dRB_tile = grad_right_tile * Rg_tile
    dLA_tile = dLB_tile * LB_tile * (1.0 - Lg_tile)
    dRA_tile = dRB_tile * RB_tile * (1.0 - Rg_tile)

    # --- Store dA and dB as bfloat16 ---
    # (This part remains the same as previous optimized version)
    tl.store(
        dLA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dLA_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dLB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dLB_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dRA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dRA_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dRB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dRB_tile,
        mask=(offs_m[:, None] < M),
    )
# fmt: on


class TritonTM1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        left_weight: torch.Tensor,
        left_gate_weight: torch.Tensor,
        right_weight: torch.Tensor,
        right_gate_weight: torch.Tensor,
    ):
        original_shape = x.shape
        x = rearrange(x, "... d -> (...) d").contiguous()
        M, N = x.shape

        left_weight = left_weight.contiguous()
        left_gate_weight = left_gate_weight.contiguous()
        right_weight = right_weight.contiguous()
        right_gate_weight = right_gate_weight.contiguous()
        left = torch.empty_like(x)
        right = torch.empty_like(x)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate_fwd_kernel[grid](
            x, left_gate_weight, left_weight, right_gate_weight, right_weight,
            left, right,
            M, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            left_weight,
            left_gate_weight,
            right_weight,
            right_gate_weight,
        )
        ctx.original_shape = original_shape

        left = left.reshape(original_shape)
        right = right.reshape(original_shape)
        return left, right

    @staticmethod
    def backward(ctx, grad_out_left: torch.Tensor, grad_out_right: torch.Tensor):
        x, left_weight, left_gate_weight, right_weight, right_gate_weight = (
            ctx.saved_tensors
        )
        x = x.to(torch.float32)
        original_shape = ctx.original_shape
        M, N = x.shape

        grad_out_left = rearrange(grad_out_left, "... d -> (...) d").contiguous()
        grad_out_right = rearrange(grad_out_right, "... d -> (...) d").contiguous()

        dLA = torch.empty_like(x)
        dLB = torch.empty_like(x)
        dRA = torch.empty_like(x)
        dRB = torch.empty_like(x)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate_bwd_kernel[grid](
            x, left_gate_weight, left_weight, right_gate_weight, right_weight,
            grad_out_left, grad_out_right, dLA, dLB, dRA, dRB,
            M, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        dx = dLA @ left_gate_weight.T + dLB @ left_weight.T
        dx += dRA @ right_gate_weight.T + dRB @ right_weight.T
        dx = dx.reshape(original_shape)

        x_t = x.T
        dWLg = torch.matmul(x_t, dLA)
        dWL = torch.matmul(x_t, dLB)
        dWRg = torch.matmul(x_t, dRA)
        dWR = torch.matmul(x_t, dRB)
        return dx, dWL, dWLg, dWR, dWRg


triton_tm1 = TritonTM1Function.apply
