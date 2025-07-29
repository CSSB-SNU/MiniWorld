import torch
import pytest

from team_gm.modules import Pairformer
from team_gm.modules.pairformer import PairformerBlock


SHAPES = [32, 64, 128, 256, 384]


@pytest.fixture(params=SHAPES)
def setup(request):
    M = request.param
    N = 128
    B = 1
    device = torch.device("cuda")
    suffix = f"[{M}, {M}, {N}]"

    rtol = 5e-3
    atol = 5e-3

    torch_model = PairformerBlock(
        Pairformer.Config(
            d_single=N, implementation="pytorch", p_drop=0, use_single_cond=True
        ),
    ).to(device)
    triton_model = PairformerBlock(
        Pairformer.Config(
            d_single=N, implementation="triton", p_drop=0, use_single_cond=True
        ),
    ).to(device)

    with torch.no_grad():
        torch.nn.init.normal_(torch_model.tri_atten_ending.out_weight)
        torch.nn.init.normal_(torch_model.tri_atten_starting.out_weight)
        torch.nn.init.normal_(torch_model.tri_multi_incoming.out_weight)
        torch.nn.init.normal_(torch_model.tri_multi_outgoing.out_weight)
        torch.nn.init.normal_(torch_model.transition_pair.squeeze_weight)
        torch.nn.init.normal_(torch_model.single_to_pair.to_out.weight)
        torch.nn.init.normal_(torch_model.pair_to_single.to_out.weight)
        torch.nn.init.normal_(torch_model.transition_single.squeeze_weight)
        triton_model.load_state_dict(torch_model.state_dict())

    single = torch.randn(B, M, N, device=device).requires_grad_()
    pair = torch.randn(B, M, M, N, device=device).requires_grad_()
    mask = torch.rand(B, M, device=device)
    mask = mask > 0.5

    pair_torch, single_torch = torch_model.forward(pair, single, mask)
    pair_triton, single_triton = triton_model.forward(pair, single, mask)

    return (
        pair,
        single,
        pair_triton,
        single_triton,
        pair_torch,
        single_torch,
        suffix,
        rtol,
        atol,
    )


def test_forward(setup):
    (
        pair,
        single,
        pair_triton,
        single_triton,
        pair_torch,
        single_torch,
        suffix,
        rtol,
        atol,
    ) = setup
    try:
        torch.testing.assert_close(pair_triton, pair_torch, rtol=rtol, atol=atol)
        torch.testing.assert_close(single_triton, single_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match forward pass for {suffix}") from e


def test_backward(setup):
    (
        pair,
        single,
        pair_triton,
        single_triton,
        pair_torch,
        single_torch,
        suffix,
        rtol,
        atol,
    ) = setup

    dy = 0.1 * torch.randn_like(pair_triton)
    pair.grad = None
    pair_torch.backward(dy, retain_graph=True)
    pair_grad_torch = pair.grad.clone()

    pair.grad = None
    pair_triton.backward(dy, retain_graph=True)
    pair_grad_triton = pair.grad.clone()

    dy = 0.1 * torch.randn_like(single_triton)
    single.grad = None
    single_torch.backward(dy, retain_graph=True)
    single_grad_torch = single.grad.clone()

    single.grad = None
    single_triton.backward(dy, retain_graph=True)
    single_grad_triton = single.grad.clone()

    try:
        torch.testing.assert_close(
            pair_grad_torch, pair_grad_triton, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            single_grad_torch, single_grad_triton, rtol=rtol, atol=atol
        )

    except AssertionError as e:
        raise AssertionError(f"❌ Results don't match backward pass for {suffix}") from e


if __name__ == "__main__":
    dtype = torch.float32
    device = torch.device("cuda")
    torch.cuda.memory_allocated(device)

    L = 384
    d_pair = 128
    pair = torch.randn(1, L, L, d_pair, device=device, dtype=dtype).requires_grad_()
    single = torch.randn(1, L, 384, device=device, dtype=dtype).requires_grad_()
    mask = torch.rand(1, L).to(device, dtype) > 0.5

    K = 52

    # config = Pairformer.Config(n_block=K)
    # pairformer = Pairformer(config).to(device, dtype)
    # y_pair, y_single = pairformer.forward(pair, single, mask)
    # y_pair.backward(torch.randn_like(y_pair))

    config = Pairformer.Config(n_block=K, implementation="triton")

    pairformer = Pairformer(config).to(device, dtype)
    y_pair, y_single = pairformer.forward(pair, single, mask)
    y_pair.backward(torch.randn_like(y_pair))

    raise ValueError(
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )
