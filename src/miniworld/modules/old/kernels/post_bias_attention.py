import torch
import triton
import triton.language as tl
import os

from einops import rearrange


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0") == "tri_attention"

if AUTOTUNE:
    configs = []
    for BM in [16, 32, 64, 128, 256]:
        for s in [1, 2, 3, 4]:
            for w in [4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": BM},
                        num_stages=s,
                        num_warps=w,
                    )
                )
else:
    configs = [
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=3),
    ]


def get_seq_group(L):
    GROUP_LENGTHS = [32, 64, 128, 256, 512]
    for group_l in GROUP_LENGTHS:
        if L <= group_l:
            return group_l
    return GROUP_LENGTHS[-1]


# fmt: off
@triton.autotune(configs=configs, key=["GROUP_M", "n_elements"])
@triton.jit
def sigmoid_gate_fwd_kernel(
    gate_ptr, rep_ptr,
    stride_gate, stride_rep,
    out_ptr,
    n_elements, R: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    out = sigmoid(gate) * rep
    Flattened 1D arrays for gate, rep, and out.
    """
    row = tl.program_id(0)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    offset_col = tl.arange(0, BLOCK_N)
    row_mask = offset_row < n_elements
    col_mask = offset_col < R
    # offset_gate = offset_row[:, None] * stride_gate + offset_col[None, :]
    offset = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)  # (...,d)
    rep = tl.load(rep_ptr + offset, mask=mask)  # (..., d)

    # out = s * rep, where s = sigmoid(gate)
    # out = s * rep, where s = sigmoid(gate)
    # s = (1.0 + tl.exp(-gate)) # (...)
    LOG2e = 1.44269504
    s = 1.0 + tl.math.exp2(-LOG2e * gate)  # (...)
    out_val = rep / s  # (..., d)

    tl.store(out_ptr + offset, out_val, mask=mask)
# fmt: on


# fmt: off
@triton.autotune(configs=configs, key=["GROUP_M", "n_elements"])
@triton.jit
def sigmoid_gate_bwd_kernel(
    gate_ptr, rep_ptr,
    grad_out_ptr, dgate_ptr, drep_ptr,
    stride_gate, stride_rep,
    n_elements, R,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Backward pass for out = sigmoid(gate) * rep
    Let s = sigmoid(gate).
    d(out)/d(gate) = rep * s*(1 - s)
    d(out)/d(rep)  = s
    """
    row = tl.program_id(0)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    offset_col = tl.arange(0, BLOCK_N)
    row_mask = offset_row < n_elements
    col_mask = offset_col < R
    # offset_gate = offset_row[:, None] * stride_gate + offset_col[None, :]
    offset = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)  # (...)
    rep = tl.load(rep_ptr + offset, mask=mask)  # (..., d)
    grad_out = tl.load(grad_out_ptr + offset, mask=mask)

    LOG2e = 1.44269504
    s = 1.0 / (1.0 + tl.math.exp2(-LOG2e * gate))  # (...)

    dgate_val = grad_out * (rep * s * (1 - s))

    # drep  = grad_out * s
    drep_val = grad_out * s

    tl.store(dgate_ptr + offset, dgate_val, mask=mask)
    tl.store(drep_ptr + offset, drep_val, mask=mask)
# fmt: on


class TritonPostBiasAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        rep: torch.Tensor,
        out_weight: torch.Tensor,
    ):
        if gate.shape != rep.shape:
            raise ValueError(f"Mismatched leading dims: {gate.shape=}, {rep.shape=}")
        if gate.ndim != 4:
            raise ValueError(f"Expected 4D gate, got {gate.ndim}D")
        op_dtype = gate.dtype

        B, L1, L2, D = gate.shape
        if out_weight.shape[0] != D:
            raise ValueError(f"Expected input dimension {D}, got {out_weight.shape[0]=}")

        gate = rearrange(gate, "... D -> (...) D").contiguous()
        rep = rearrange(rep, "... D -> (...) D").contiguous()
        _out_weight = out_weight.to(gate.dtype)
        M, N = rep.shape
        out = torch.empty_like(rep)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        sigmoid_gate_fwd_kernel[grid](
            gate, rep,
            gate.stride(0), rep.stride(0),
            out,
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        ctx.save_for_backward(
            gate.to(torch.bfloat16),
            rep.to(torch.bfloat16),
            out_weight,
        )
        ctx.orig_shape = (B, L1, L2, D)
        ctx.op_dtype = op_dtype

        out = torch.matmul(out, _out_weight)
        out = rearrange(out, "(B L1 L2) W -> B L1 L2 W", B=B, L1=L1, L2=L2)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, rep, out_weight = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        gate = gate.to(op_dtype)
        rep = rep.to(op_dtype)
        _out_weight = out_weight.to(op_dtype)
        grad_out = grad_out.to(op_dtype)
        orig_shape = ctx.orig_shape
        M, N = rep.shape
        out = torch.empty_like(rep)

        # fmt: off
        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        sigmoid_gate_fwd_kernel[grid](
            gate, rep,
            gate.stride(0), rep.stride(0),
            out,
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        grad_out = rearrange(grad_out, "... W -> (...) W").contiguous()
        dW_out = torch.matmul(out.T, grad_out)
        grad_out = torch.matmul(grad_out, _out_weight.T)

        dgate = torch.empty_like(gate)
        drep = torch.empty_like(rep)

        # fmt: off
        sigmoid_gate_bwd_kernel[grid](
            gate, rep,
            grad_out, dgate, drep,
            gate.stride(0), rep.stride(0),
            M, N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )
        # fmt: on

        dgate = dgate.reshape(orig_shape)
        drep = drep.reshape(orig_shape)
        return dgate, drep, dW_out.to(torch.float32)


triton_post_bias_attention = TritonPostBiasAttentionFunction.apply
