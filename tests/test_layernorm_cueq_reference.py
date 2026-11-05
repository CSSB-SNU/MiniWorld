"""``miniworld.layers.primitives.LayerNorm`` routes cuEquivariance to the PyTorch reference (moved from FoldForge's test_inference_backends)."""

import copy

import pytest
import torch
from foldforge.layers.implementation import ImplementationType


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [True, False])
def test_cueq_norm_uses_reference_with_fp32_affine(dtype, affine):
    from miniworld.layers.primitives import LayerNorm

    norm = LayerNorm(
        32, elementwise_affine=affine, implementation=ImplementationType.CUEQUIVARIANCE
    ).to(dtype)
    x = torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)
    reference = copy.deepcopy(norm)
    reference.implementation = ImplementationType.PYTORCH
    ref_x = x.detach().clone().requires_grad_()
    output = norm(x)
    expected = reference(ref_x)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.dtype == dtype
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    if affine:
        assert norm.weight.dtype == torch.float32
        torch.testing.assert_close(
            norm.weight.grad, reference.weight.grad, atol=0, rtol=0
        )
