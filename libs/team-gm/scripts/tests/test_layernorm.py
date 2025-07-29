import torch
import pytest

from team_gm.modules.primitives import LayerNorm

SHAPES = [
    (64, 128),
    (128, 128),
    (256, 128),
    (32, 384),
    (64, 384),
]


@pytest.fixture(params=SHAPES)
def setup(request):
    M, N = request.param
    eps = 1e-5
    device = torch.device("cuda")
    suffix = f"[{M}, {M}, {N}]"

    rtol = 5e-3
    atol = 5e-3

    x_shape = (M * M, N)
    w_shape = (x_shape[-1],)
    triton_model = LayerNorm(w_shape, eps, True, implementation="triton").to(device)
    torch_model = LayerNorm(w_shape, eps, True, implementation="pytorch").to(device)

    with torch.no_grad():
        triton_model.load_state_dict(torch_model.state_dict())

    x = torch.randn(x_shape, device=device).requires_grad_(True)

    y_triton = triton_model(x)
    y_torch = torch_model(x)
    return x, y_triton, y_torch, suffix, rtol, atol


def test_forward(setup):
    x, y_triton, y_torch, suffix, rtol, atol = setup
    try:
        torch.testing.assert_close(y_triton, y_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    x, y_triton, y_torch, suffix, rtol, atol = setup
    dy = 0.1 * torch.randn_like(x)

    x.grad = None
    y_triton.backward(dy, retain_graph=True)
    grad_triton = x.grad.clone()

    x.grad = None
    y_torch.backward(dy, retain_graph=True)
    grad_torch = x.grad.clone()

    try:
        torch.testing.assert_close(grad_triton, grad_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match backward pass for {suffix}") from e
