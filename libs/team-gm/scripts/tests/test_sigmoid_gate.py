import torch
import pytest

from team_gm.modules.primitives import SigmoidGateFunction

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
    device = torch.device("cuda")
    suffix = f"[{M}, {M}, {N}]"

    rtol = 5e-3
    atol = 5e-3

    triton_model = SigmoidGateFunction("triton").to(device)
    torch_model = SigmoidGateFunction("pytorch").to(device)

    gate = torch.randn(M, M, N, device=device).requires_grad_(True)
    rep = torch.randn(M, M, N, device=device).requires_grad_(True)

    y_triton = triton_model(gate, rep)
    y_torch = torch_model(gate, rep)
    return gate, rep, y_triton, y_torch, suffix, rtol, atol


def test_forward(setup):
    gate, rep, y_triton, y_torch, suffix, rtol, atol = setup
    try:
        torch.testing.assert_close(y_triton, y_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    gate, rep, y_triton, y_torch, suffix, rtol, atol = setup
    dy = 0.1 * torch.randn_like(rep)

    gate.grad = None
    rep.grad = None
    y_triton.backward(dy, retain_graph=True)
    gate_grad_triton = gate.grad.clone()
    rep_grad_triton = rep.grad.clone()

    gate.grad = None
    rep.grad = None
    y_torch.backward(dy, retain_graph=True)
    gate_grad_torch = gate.grad.clone()
    rep_grad_torch = rep.grad.clone()

    try:
        torch.testing.assert_close(
            gate_grad_triton, gate_grad_torch, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(rep_grad_triton, rep_grad_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match backward pass for {suffix}") from e
