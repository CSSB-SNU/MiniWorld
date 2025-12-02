import torch
import triton
import triton.language as tl

from einops import rearrange


# fmt: off
@triton.jit
def sigmoid_gate_g1_fwd_kernel(
    gate_ptr, rep_ptr,
    stride_gate, stride_rep,
    out_ptr,
    n_elements, R,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    offset_gate = offset_row * stride_gate
    offset_rep = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask_rep = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset_gate, mask=row_mask).to(tl.float32)  # (...)
    rep = tl.load(rep_ptr + offset_rep, mask=mask_rep)  # (..., d)
    LOG2e = 1.44269504

    # out = s * rep, where s = sigmoid(gate)
    # s = (1.0 + tl.exp(-gate)) # (...)
    s = 1.0 + tl.math.exp2(-LOG2e * gate)  # (...)
    out_val = rep / s[:, None]  # (..., d)
    out_val = out_val.to(tl.bfloat16)

    tl.store(out_ptr + offset_rep, out_val, mask=mask_rep)
# fmt: on


# fmt: off
@triton.jit
def sigmoid_gate_gr_fwd_kernel(
    gate_ptr, rep_ptr,
    stride_gate, stride_rep,
    out_ptr,
    n_elements, R,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    out_val = out_val.to(tl.bfloat16)

    tl.store(out_ptr + offset, out_val, mask=mask)
# fmt: on


# fmt: off
@triton.jit
def sigmoid_gate_g1_bwd_kernel(
    gate_ptr, rep_ptr, grad_out_ptr, dgate_ptr, drep_ptr,
    stride_gate, stride_rep,
    n_elements, R,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    offset_gate = offset_row * stride_gate
    offset_rep = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask_rep = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset_gate, mask=row_mask).to(tl.float32)  # (...)
    rep = tl.load(rep_ptr + offset_rep, mask=mask_rep)  # (..., d)
    grad_out = tl.load(grad_out_ptr + offset_rep, mask=mask_rep)

    LOG2e = 1.44269504
    s = 1.0 / (1.0 + tl.math.exp2(-LOG2e * gate))  # (...)

    s = s[:, None]  # (..., 1)

    dgate_val = grad_out * (rep * s * (1 - s))
    dgate_val = tl.sum(dgate_val, axis=-1)  # sum over last dim

    # drep  = grad_out * s
    drep_val = grad_out * s

    dgate_val = dgate_val.to(tl.bfloat16)
    drep_val = drep_val.to(tl.bfloat16)

    tl.store(dgate_ptr + offset_gate, dgate_val, mask=row_mask)
    tl.store(drep_ptr + offset_rep, drep_val, mask=mask_rep)
# fmt: on


# fmt: off
@triton.jit
def sigmoid_gate_gr_bwd_kernel(
    gate_ptr, rep_ptr, grad_out_ptr, dgate_ptr, drep_ptr,
    stride_gate, stride_rep,
    n_elements, R,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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

    dgate_val = dgate_val.to(tl.bfloat16)
    drep_val = drep_val.to(tl.bfloat16)

    tl.store(dgate_ptr + offset, dgate_val, mask=mask)
    tl.store(drep_ptr + offset, drep_val, mask=mask)
# fmt: on


class TritonSigmoidGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, rep: torch.Tensor):
        if gate.shape[:-1] != rep.shape[:-1]:
            raise ValueError(f"Mismatched leading dims: {gate.shape=}, {rep.shape=}")

        op_dtype = gate.dtype
        rest_dims = gate.shape[:-1]
        G = gate.shape[-1]
        R = rep.shape[-1]

        if G not in (1, R):
            raise ValueError(f"Last dim of gate must be 1 or {R}, but {gate.shape=}")
        broadcasted = G == 1

        gate = rearrange(gate, "... G -> (...) G").contiguous()
        rep = rearrange(rep, "... R -> (...) R").contiguous()
        n_elements = rep.shape[0]
        out = torch.empty_like(rep)

        if broadcasted:
            kernel = sigmoid_gate_g1_fwd_kernel
        else:
            kernel = sigmoid_gate_gr_fwd_kernel

        # fmt: off
        grid = lambda META: [triton.cdiv(n_elements, META["BLOCK_M"])]
        kernel[grid](
            gate, rep,
            gate.stride(0), rep.stride(0),
            out,
            n_elements, R,
            BLOCK_M=128, BLOCK_N=triton.next_power_of_2(R),
            num_warps=8,
        )
        # fmt: on

        ctx.save_for_backward(gate.bfloat16(), rep.bfloat16())
        ctx.rest_dims = rest_dims
        ctx.op_dtype = op_dtype
        out = out.view(*rest_dims, R)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, rep = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        gate, rep = gate.to(op_dtype), rep.to(op_dtype)
        rest_dims = ctx.rest_dims
        G = gate.shape[-1]
        R = rep.shape[-1]
        broadcasted = G == 1

        gate = rearrange(gate, "... G -> (...) G").contiguous()
        rep = rearrange(rep, "... R -> (...) R").contiguous()
        grad_out = rearrange(grad_out, "... R -> (...) R").contiguous()
        n_elements = rep.shape[0]
        dgate = torch.empty_like(gate)
        drep = torch.empty_like(rep)

        if broadcasted:
            kernel = sigmoid_gate_g1_bwd_kernel
        else:
            kernel = sigmoid_gate_gr_bwd_kernel

        # fmt: off
        grid = lambda META: [triton.cdiv(n_elements, META["BLOCK_M"])]
        kernel[grid](
            gate, rep, grad_out, dgate, drep,
            gate.stride(0), rep.stride(0),
            n_elements, R,
            BLOCK_M=128, BLOCK_N=triton.next_power_of_2(R),
            num_warps=8,
        )
        # fmt: on

        dgate = dgate.view(*rest_dims, G)
        drep = drep.view(*rest_dims, R)
        return dgate, drep


triton_sigmoid_gate = TritonSigmoidGateFunction.apply
