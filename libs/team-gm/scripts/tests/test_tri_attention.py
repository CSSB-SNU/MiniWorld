import torch
import pytest

from team_gm.modules.attentions import TriangleAttention


SHAPES = [32, 64, 128, 256, 384]

STARTING = [True, False]

SELF_ATTENTION = [True, False]


@pytest.fixture(
    params=[
        (M, starting, use_self_attention)
        for M in SHAPES
        for starting in STARTING
        for use_self_attention in SELF_ATTENTION
    ]
)
def setup(request):
    M, starting, use_self_attention = request.param
    N = 128
    B = 1
    device = torch.device("cuda")
    suffix = f"[{M}, {M}, {N}], {starting=}, {use_self_attention=}"

    rtol = 5e-2
    atol = 5e-2

    torch_model = TriangleAttention(
        d_pair=N,
        starting=starting,
        implementation="pytorch",
        use_self_attention=use_self_attention,
    ).to(device)
    triton_model = TriangleAttention(
        d_pair=N,
        starting=starting,
        implementation="triton",
        use_self_attention=use_self_attention,
    ).to(device)

    with torch.no_grad():
        torch.nn.init.normal_(torch_model.out_weight)
        triton_model.load_state_dict(torch_model.state_dict())

    pair = torch.randn(B, M, M, N, device=device).requires_grad_()
    mask = torch.rand(B, M, device=device)
    mask = mask > 0.5

    y_torch = torch_model.forward(pair, mask)
    y_triton = triton_model.forward(pair, mask)
    return pair, y_triton, y_torch, suffix, rtol, atol


def test_forward(setup):
    pair, y_triton, y_torch, suffix, rtol, atol = setup
    try:
        torch.testing.assert_close(y_torch, y_triton, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    pair, y_triton, y_torch, suffix, rtol, atol = setup

    dy = 0.1 * torch.randn_like(y_triton)

    pair.grad = None
    y_torch.backward(dy, retain_graph=True)
    pair_grad_torch = pair.grad.clone()

    pair.grad = None
    y_triton.backward(dy, retain_graph=True)
    pair_grad_triton = pair.grad.clone()
    try:
        torch.testing.assert_close(
            pair_grad_torch, pair_grad_triton, rtol=rtol, atol=atol
        )

    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match backward pass for {suffix}") from e


if __name__ == "__main__":
    import torch.nn as nn

    class TestModule(nn.Module):
        def __init__(self, k=1, implementation="pytorch"):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TriangleAttention(implementation=implementation, starting=True)
                    for _ in range(k)
                ]
            )

        def forward(self, pair, mask):
            for layer in self.layers:
                pair = layer(pair, mask)
            return pair

    k = 40
    device = torch.device("cuda")
    dtype = torch.float32

    L = 384
    N = 128
    B = 1
    pair = torch.randn(B, L, L, N).to(device, dtype).requires_grad_()
    mask = torch.rand(B, L).to(device, dtype) > 0.5

    # torch_model = TestModule(implementation="pytorch", k=k).to(device, dtype)
    # y_torch = torch_model.forward(pair, mask)
    # y_torch.backward(torch.randn_like(y_torch))

    triton_model = TestModule(implementation="triton", k=k).to(device, dtype)
    y_triton = triton_model.forward(pair, mask)
    y_triton.backward(torch.randn_like(y_triton))
