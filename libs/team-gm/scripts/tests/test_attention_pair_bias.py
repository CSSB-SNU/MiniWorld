import torch
import pytest

from team_gm.modules.attentions import AttentionPairBias

SHAPES = [
    # (B, L, H, S, P)
    (1, 32, 12, 384, 128),
    (1, 64, 16, 384, 128),
    (1, 256, 12, 384, 128),
    (1, 384, 12, 384, 256),
    (1, 512, 8, 256, 128),
    (5, 32, 12, 384, 128),
    (5, 64, 16, 384, 128),
    (5, 256, 12, 384, 128),
    (5, 384, 12, 384, 256),
    (5, 512, 8, 256, 128),
]


@pytest.fixture(params=SHAPES)
def setup(request):
    B, L, H, S, P = request.param
    device = torch.device("cuda")
    suffix = f"B={B}, L={L}, H={H}, d={S}/{P}"

    rtol = 5e-3
    atol = 5e-3

    torch_model = AttentionPairBias(S, P, H, implementation="pytorch").to(device)
    triton_model = AttentionPairBias(S, P, H, implementation="triton").to(device)

    with torch.no_grad():
        torch.nn.init.normal_(torch_model.to_out.weight)
        triton_model.load_state_dict(torch_model.state_dict())

    single = torch.randn(B, L, S, device=device).requires_grad_()
    pair = torch.randn(B, L, L, P, device=device).requires_grad_()
    mask = torch.rand(B, L, device=device) >= 0.5

    y_torch = torch_model(single, pair, mask)
    y_triton = triton_model(single, pair, mask)
    return single, pair, y_triton, y_torch, suffix, rtol, atol


def test_forward(setup):
    single, pair, y_triton, y_torch, suffix, rtol, atol = setup
    try:
        torch.testing.assert_close(y_torch, y_triton, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    single, pair, y_triton, y_torch, suffix, rtol, atol = setup
    dy = 0.1 * torch.randn_like(y_triton)

    single.grad = None
    pair.grad = None
    y_torch.backward(dy, retain_graph=True)
    single_grad_torch = single.grad.clone()
    pair_grad_torch = pair.grad.clone()

    single.grad = None
    pair.grad = None
    y_triton.backward(dy, retain_graph=True)
    single_grad_triton = single.grad.clone()
    pair_grad_triton = pair.grad.clone()

    try:
        torch.testing.assert_close(
            single_grad_torch, single_grad_triton, rtol=rtol, atol=atol
        )
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
                [AttentionPairBias(implementation=implementation) for _ in range(k)]
            )

        def forward(self, single, pair, mask):
            for layer in self.layers:
                single = layer(single, pair, mask)
            return pair

    k = 50
    device = torch.device("cuda")
    dtype = torch.bfloat16

    L = 1024
    N = 128
    B = 1
    single = torch.randn(B, L, 384).to(device, dtype).requires_grad_()
    pair = torch.randn(B, L, L, N).to(device, dtype).requires_grad_()
    mask = torch.rand(B, L).to(device, dtype) > 0.5

    torch_model = TestModule(implementation="pytorch", k=k).to(device, dtype)
    y_torch = torch_model.forward(single, pair, mask)
    y_torch.backward(torch.randn_like(y_torch))

    # triton_model = TestModule(implementation="triton", k=k).to(device, dtype)
    # y_triton = triton_model.forward(single, pair, mask)
    # y_triton.backward(torch.randn_like(y_triton))
