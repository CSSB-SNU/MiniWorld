"""Numerical and gradient coverage for the remaining engine connections."""

import copy

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference_backends(module):
    from miniworld_engine.modules.dispatch import KernelBackend
    from team_gm.modules.exceptions import ImplementationType

    reference = copy.deepcopy(module)
    for child in reference.modules():
        if hasattr(child, "_backend"):
            child._backend = KernelBackend.PYTORCH
        if type(child).__name__ in {"SWAAtomBlock", "SwiGLUFFN"}:
            child.implementation = ImplementationType.PYTORCH
    return reference


def check_outputs_and_gradients(native, reference, inputs, *, output_tol=0.03):
    refs = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = native(*inputs)
        expected = reference(*refs)
    assert actual.dtype == expected.dtype
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert torch.isfinite(actual).all() and error < output_tol, error
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    pairs = [("input", x.grad, r.grad) for x, r in zip(inputs, refs) if x.requires_grad]
    pairs += [
        (name, p.grad, r.grad)
        for (name, p), r in zip(native.named_parameters(), reference.parameters())
        if p.requires_grad
    ]
    for name, actual_grad, expected_grad in pairs:
        assert actual_grad is not None and expected_grad is not None, name
        assert torch.isfinite(actual_grad).all(), name
        if name == "ln_pair.bias":
            # A key-independent attention offset has zero exact gradient.
            scale = reference.ln_pair.weight.grad.float().norm()
            assert actual_grad.float().norm() < torch.finfo(torch.bfloat16).eps * scale
            continue
        error = (
            actual_grad.float() - expected_grad.float()
        ).norm() / expected_grad.float().norm().clamp_min(1e-8)
        assert error < 0.08, (name, error)


@pytest.mark.parametrize("width", [48, 128])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_keeps_epsilon_dtype_and_gradients(width, affine, dtype):
    from miniworld_engine.modules import RMSNorm

    torch.manual_seed(18)
    native = RMSNorm(width, elementwise_affine=affine, implementation="miniworld").cuda()
    reference = torch.nn.RMSNorm(width, elementwise_affine=affine).cuda()
    reference.load_state_dict(native.state_dict(), strict=True)
    x = (torch.randn(2, 128, width, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    # Small values catch input-dtype epsilon instead of torch's accumulation-dtype epsilon.
    check_outputs_and_gradients(native, reference, [x])


@pytest.mark.parametrize("variant", ["esmfold2", "af3"])
def test_swa_atom_block_uses_modulation_ffn_and_preserves_checkpoint(variant):
    from team_gm.modules.blocks.swa_atom_transformer import SWAAtomBlock
    from team_gm.modules.blocks.rope_swa_af3_transformer import RoPESWAAF3Block
    from team_gm.modules.exceptions import ImplementationType

    torch.manual_seed(19)
    np.random.seed(19)
    block_type = SWAAtomBlock if variant == "esmfold2" else RoPESWAAF3Block
    native = block_type(
        128, 128, 4, implementation=ImplementationType.MINIWORLD_ENGINE
    ).cuda()
    with torch.no_grad():
        if variant == "esmfold2":
            native.adaln_modulation[1].weight.normal_(std=0.02)
        else:
            native.attention.to_out.weight.normal_(std=0.02)
            native.transition.squeeze.weight.normal_(std=0.02)
    reference = reference_backends(native)
    reference.load_state_dict(native.state_dict(), strict=True)
    n, length = 2, 128
    valid = torch.ones(n, length, device="cuda", dtype=torch.bool)
    valid[:, -8:] = False
    params = (
        torch.ones(n, length, 16, device="cuda"),
        torch.zeros(n, length, 16, device="cuda"),
        valid.sum(-1).int(),
        torch.arange(n + 1, device="cuda", dtype=torch.int32) * length,
        length,
        valid,
    )

    class Bound(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, q, c):
            return self.block(q, c, params)

    q = torch.randn(
        n, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    c = torch.randn(
        n, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    check_outputs_and_gradients(Bound(native), Bound(reference), [q, c])


@pytest.mark.parametrize("qk_norm", [False, True])
def test_confidence_attention_engine_core(qk_norm):
    from miniworld_engine.modules import AttentionPairBias

    torch.manual_seed(20)
    np.random.seed(20)
    native = AttentionPairBias(
        384, 128, 16, use_qk_norm=qk_norm, implementation="miniworld"
    ).cuda()
    with torch.no_grad():
        native.to_out.weight.normal_(std=0.02)
    reference = reference_backends(native)
    x = torch.randn(2, 128, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    p = torch.randn(
        2, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.ones(2, 128, device="cuda", dtype=torch.bool)
    mask[:, -8:] = False
    mask[1] = False
    check_outputs_and_gradients(native, reference, [x, p, mask])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_confidence_head_backward_and_native_selection(batch_size):
    from miniworld.modules.confidence_head import ConfidenceHead
    from miniworld_engine.modules.dispatch import KernelBackend
    from team_gm.modules.exceptions import ImplementationType

    torch.manual_seed(22)
    np.random.seed(22)
    head = (
        ConfidenceHead(
            ConfidenceHead.Config(
                d_single_input=449,
                n_block=4,
                implementation=ImplementationType.MINIWORLD_ENGINE,
            )
        )
        .cuda()
        .float()
        .train()
    )
    for block in head.pairformer.blocks:
        assert block.attention_pair_bias._backend == KernelBackend.TRITON
    with torch.no_grad():
        for layer in (head.to_pae, head.to_pde, head.to_plddt):
            layer.weight.normal_(std=0.02)
        # Exercise each residual branch rather than its zero-initialized identity.
        for block in head.pairformer.blocks:
            for layer in (
                block.tri_multi.to_out,
                block.transition_pair.squeeze,
                block.attention_pair_bias.to_out,
                block.transition_single.squeeze,
            ):
                layer.weight.normal_(std=0.02)
    length = 128
    inputs = (
        torch.randn(batch_size, length, length, 128, device="cuda"),
        torch.randn(batch_size, length, 449, device="cuda"),
        torch.rand(batch_size, length, length, device="cuda") * 20,
        torch.ones(batch_size, length, device="cuda", dtype=torch.bool),
        torch.arange(length, device="cuda")[None].expand(batch_size, -1),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = head(*inputs)
        loss = sum(v.float().square().mean() for v in out.values())
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    head.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference = head(*inputs)
    for name, value in inference.items():
        assert torch.isfinite(value).all(), name
        error = (value.float() - out[name].float()).norm() / out[name].float().norm()
        assert error < 0.03, (name, error)
