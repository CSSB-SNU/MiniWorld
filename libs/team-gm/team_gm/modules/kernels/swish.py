import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def fused_swish_fwd_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load inputs
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask)

    # compute sigmoid(x)
    # For large negative x, e^-x can overflow. In practice, you might clamp x.
    sigx = 1.0 / (1.0 + tl.exp(-x))

    # swish: out = x * sigx * y
    out_val = x * sigx * y
    out_val = out_val.to(tl.bfloat16)

    # store output
    tl.store(out_ptr + offsets, out_val, mask=mask)
# fmt: on


# fmt: off
@triton.jit
def fused_swish_bwd_kernel(
    x_ptr, y_ptr, grad_out_ptr, dx_ptr, dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)

    # s = sigmoid(x)
    s = 1.0 / (1.0 + tl.exp(-x))

    # dx = grad_out * [y * s + y * x * s (1 - s)]
    dx_val = grad_out * (y * s + y * x * s * (1 - s))

    # dy = grad_out * [x * s]
    dy_val = grad_out * (x * s)

    dx_val = dx_val.to(tl.bfloat16)
    dy_val = dy_val.to(tl.bfloat16)

    # store
    tl.store(dx_ptr + offsets, dx_val, mask=mask)
    tl.store(dy_ptr + offsets, dy_val, mask=mask)
# fmt: on


class TritonSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        x = x.contiguous()
        y = y.contiguous()

        out = torch.empty_like(x)
        n_elements = x.numel()

        # fmt: off
        grid = lambda META: [triton.cdiv(n_elements, META["BLOCK_SIZE"])]
        fused_swish_fwd_kernel[grid](
            x, y, out,
            n_elements,
            BLOCK_SIZE=1024,
        )
        # fmt: on

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            y.to(torch.bfloat16),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        grad_output = grad_output.contiguous()

        dx = torch.empty_like(x)
        dy = torch.empty_like(y)
        n_elements = x.numel()

        # fmt: off
        grid = lambda META: [triton.cdiv(n_elements, META["BLOCK_SIZE"])]
        fused_swish_bwd_kernel[grid](
            x, y, grad_output, dx, dy,
            n_elements,
            BLOCK_SIZE=1024,
        )
        # fmt: on

        return dx, dy


triton_swish = TritonSwishFunction.apply
