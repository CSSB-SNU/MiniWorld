import torch
import triton
import triton.language as tl
import os

from einops import rearrange

from .utils import get_seq_group
from .layernorm import (
    layer_norm_fwd_fused,
    layer_norm_bwd_dx_fused,
    layer_norm_fwd_fused_recal,
)


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"


if AUTOTUNE:
    fwd_configs = []
    for BM in [16, 32, 64]:
        for BK in [16, 32, 64]:
            if BM * BK > 2048:
                continue
            for s in [1, 2, 3]:
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
            for s in [1, 2, 3]:
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
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ]


# fmt: off
@triton.autotune(configs=fwd_configs, key=["GROUP_M", "d"])
@triton.jit
def fused_sigmoid_gate2_fwd_kernel(
    x_gate_ptr, x_out_ptr, W_gate_ptr, W_out_ptr, out_ptr,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_d_out = tl.arange(0, N)

    A_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_gate_tile = tl.load(
            x_gate_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        x_out_tile = tl.load(
            x_out_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W_gate_tile = tl.load(
            W_gate_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile, allow_tf32=False)
        B_tile += tl.dot(x_out_tile, W_out_tile, allow_tf32=False)

    g_tile = tl.sigmoid(A_tile)
    out_tile = g_tile * B_tile
    out_ptr_ = out_ptr + (offs_m[:, None] * N + offs_d_out[None, :])
    tl.store(out_ptr_, out_tile, mask=(offs_m[:, None] < M))
# fmt: on


# fmt: off
@triton.autotune(configs=bwd_configs, key=["GROUP_M", "d"])
@triton.jit
def fused_sigmoid_gate2_bwd_kernel(
    x_gate_ptr, x_out_ptr, W_gate_ptr, W_out_ptr,
    grad_out_ptr, dA_ptr, dB_ptr,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_d_full = tl.arange(0, N)

    # --- Recompute A = x_gate @ W_gate and B = x_out @ W_out in float32 ---
    A_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_gate_tile = tl.load(
            x_gate_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        x_out_tile = tl.load(
            x_out_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W_gate_tile = tl.load(
            W_gate_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile, allow_tf32=False)
        B_tile += tl.dot(x_out_tile, W_out_tile, allow_tf32=False)

    # --- Compute full dA and dB (Element-wise in float32) ---
    g_tile = tl.sigmoid(A_tile)
    grad_tile = tl.load(
        grad_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
        other=0.0,
    )
    dB_tile = grad_tile * g_tile
    dA_tile = dB_tile * (B_tile * (1.0 - g_tile))

    # --- Store full dA and dB as bfloat16 (Needed for PyTorch dW calc) ---
    tl.store(
        dA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dA_tile,  # Cast down for storage
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
    )
    tl.store(
        dB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dB_tile,  # Cast down for storage
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
    )
# fmt: on


class TritonTM2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_out: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        gate_weight: torch.Tensor,
        out_weight: torch.Tensor,
    ):
        original_shape = x.shape
        op_dtype = x.dtype
        if x.dtype != x_out.dtype:
            msg = f"x and x_out must have the same dtype, got {x.dtype} and {x_out.dtype}"
            raise ValueError(msg)
        x = rearrange(x, "... d -> (...) d").contiguous()
        x_out = rearrange(x_out, "... d -> (...) d").contiguous()
        M, N = x_out.shape

        y = torch.empty_like(x_out)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        _ln_weight = ln_weight.to(op_dtype)
        _ln_bias = ln_bias.to(op_dtype)
        _gate_weight = gate_weight.to(op_dtype).contiguous()
        _out_weight = out_weight.to(op_dtype).contiguous()

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused[grid](
            x_out, y, _ln_weight, _ln_bias, mean, rstd,
            x_out.stride(0), x_out.stride(1),
            M, N, 1e-5,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        out = torch.empty_like(x)

        # fmt: off
        fused_sigmoid_gate2_fwd_kernel[grid](
            x, y, _gate_weight, _out_weight, out,
            M, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        ctx.save_for_backward(
            x.bfloat16(),
            x_out.bfloat16(),
            ln_weight,
            ln_bias,
            gate_weight,
            out_weight,
            mean,
            rstd,
        )
        ctx.original_shape = original_shape
        ctx.op_dtype = op_dtype

        out = out.reshape(original_shape)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, x_out, ln_weight, ln_bias, gate_weight, out_weight, mean, rstd = (
            ctx.saved_tensors
        )
        op_dtype = ctx.op_dtype
        x, x_out = x.to(op_dtype), x_out.to(op_dtype)
        _gate_weight = gate_weight.to(op_dtype)
        _out_weight = out_weight.to(op_dtype)

        original_shape = ctx.original_shape
        M, N = x.shape

        y = torch.empty_like(x)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        layer_norm_fwd_fused_recal[grid](
            x_out, y, ln_weight, ln_bias, mean, rstd,
            x_out.stride(0), x_out.stride(1),
            M, N,
            GROUP_M=get_seq_group(M),
            BLOCK_N=triton.next_power_of_2(N),
        )
        # fmt: on

        grad_out = rearrange(grad_out, "... d -> (...) d").contiguous()
        dA = torch.empty_like(x)
        dB = torch.empty_like(x)

        # fmt: off
        fused_sigmoid_gate2_bwd_kernel[grid](
            x, y, _gate_weight, _out_weight,
            grad_out, dA, dB,
            M, N,
            GROUP_M=get_seq_group(M),
        )
        # fmt: on
        dx = dA @ _gate_weight.T
        dx_ln = dB @ _out_weight.T
        dW_gate = torch.matmul(x.T, dA)
        dW_out = torch.matmul(y.T, dB)

        dx_out = torch.empty_like(x)
        dw = torch.zeros(N, dtype=torch.float32, device=x.device)
        db = torch.zeros(N, dtype=torch.float32, device=x.device)

        # fmt: off
        layer_norm_bwd_dx_fused[grid](
            dx_out, dx_ln, dw, db,
            x_out, ln_weight, mean, rstd,
            dw.stride(0), db.stride(0), x_out.stride(0), x_out.stride(1),
            M, N,
            GROUP_M=get_seq_group(M),
            BLOCK_N=triton.next_power_of_2(N),
        )
        # fmt: on

        dx = dx.reshape(original_shape)
        dx_out = dx_out.reshape(original_shape)
        return dx, dx_out, dw.float(), db.float(), dW_gate.float(), dW_out.float()


triton_tm2 = TritonTM2Function.apply
