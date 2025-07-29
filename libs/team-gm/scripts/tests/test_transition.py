import torch
import pytest

from team_gm.modules.primitives import Transition

SHAPES = [
    (1, 32, 128),
    (1, 64, 128),
    (1, 128, 128),
    (1, 256, 128),
    (1, 384, 128),
    (5, 32, 128),
    (5, 64, 128),
    (5, 128, 128),
    (5, 256, 128),
    (5, 384, 128),
]


@pytest.fixture(params=SHAPES)
def setup(request):
    B, M, N = request.param
    device = torch.device("cuda")
    suffix = f"[{B}, {M}, {M}, {N}]"

    rtol = 5e-2
    atol = 5e-2

    torch_model = Transition(N, implementation="pytorch").to(device)
    triton_model = Transition(N, implementation="triton").to(device)

    with torch.no_grad():
        torch.nn.init.normal_(torch_model.squeeze_weight)
        triton_model.load_state_dict(torch_model.state_dict())

    x = torch.randn(B, M, M, N, device=device).requires_grad_(True)

    y_torch = torch_model(x)
    y_triton = triton_model(x)
    return x, y_triton, y_torch, suffix, rtol, atol


def test_forward(setup):
    x, y_triton, y_torch, suffix, rtol, atol = setup
    try:
        torch.testing.assert_close(y_torch, y_triton, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    x, y_triton, y_torch, suffix, rtol, atol = setup
    dy = 0.1 * torch.randn_like(y_triton)

    x.grad = None
    y_torch.backward(dy, retain_graph=True)
    grad_torch = x.grad.clone()

    x.grad = None
    y_triton.backward(dy, retain_graph=True)
    grad_triton = x.grad.clone()

    try:
        torch.testing.assert_close(grad_torch, grad_triton, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match backward pass for {suffix}") from e


if __name__ == "__main__":
    import torch.nn as nn

    class TestModule(nn.Module):
        def __init__(self, k, d, implementation):
            super().__init__()
            self.module_list = nn.ModuleList(
                [Transition(d, implementation=implementation) for _ in range(k)]
            )

        def forward(self, x):
            for layer in self.module_list:
                x = layer(x)
            return x

    k = 500
    device = torch.device("cuda")

    L = 384
    D = 128
    B = 1
    pair = torch.randn(B, L, L, D).to(device).requires_grad_()

    # torch_model = TestModule(implementation="pytorch", k=k, d=D).to(device)
    # y_torch = torch_model.forward(pair)
    # y_torch.backward(torch.randn_like(y_torch))

    triton_model = TestModule(implementation="triton", k=k, d=D).to(device)
    y_triton = triton_model.forward(pair)
    y_triton.backward(torch.randn_like(y_triton))
