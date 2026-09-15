"""Check newly connected engine paths against the PyTorch implementation."""

import copy

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("recycles", [1, 2, 4])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_condition_loop_fullgraph_preserves_last_recycle_grad(recycles, grad_enabled):
    from types import SimpleNamespace

    from miniworld.models.diffusion.model import DiffusionModel

    class TinyTrunk(torch.nn.Module):
        _condition_impl = DiffusionModel._condition_impl

        def __init__(self):
            super().__init__()
            self._forced_n_recycle = recycles
            self.weight = torch.nn.Parameter(torch.tensor(0.7))

        def _embed(self, msa, *args):
            return msa, msa, msa, None, None

        def _trunk_step(self, pair, initial, *args):
            return (pair * self.weight + initial).sin()

    model = TinyTrunk()
    fn = torch.compile(model._condition_impl, backend="eager", fullgraph=True)
    x = torch.tensor([0.1, 0.3, 0.6])
    with torch.set_grad_enabled(grad_enabled):
        _, actual = fn(x, None, SimpleNamespace(token_asym_id=None), None, None)
        reference = torch.zeros_like(x)
        for i in range(recycles):
            reference = (reference * model.weight + x).sin()
            if i < recycles - 1:
                reference = reference.detach()
    torch.testing.assert_close(actual, reference)
    assert actual.requires_grad == grad_enabled
    if grad_enabled:
        actual_grad = torch.autograd.grad(actual.sum(), model.weight)[0]
        expected_grad = torch.autograd.grad(reference.sum(), model.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["opm", "pwa", "pair_cond", "single_cond"])
@pytest.mark.parametrize("training", [False, True])
def test_connected_paths_amp(kind, training):
    from miniworld.configs.models import SharedConfig
    from miniworld.modules.diffusion_module import DiffusionConditioning
    from miniworld.modules.mini_msa_module import MiniMSAModuleBlock
    from miniworld_engine.modules.dispatch import KernelBackend
    from team_gm.modules.exceptions import ImplementationType

    torch.manual_seed(21)
    np.random.seed(21)
    if kind in ("opm", "pwa"):
        block = MiniMSAModuleBlock(
            p_drop=0,
            p_drop_msa=0,
            implementation=ImplementationType.MINIWORLD_ENGINE,
        )
        module = (
            block.outer_product_mean
            if kind == "opm"
            else block.msa_pair_weighted_averaging
        )
        assert module.ln_msa._backend != KernelBackend.PYTORCH
        if kind == "pwa":
            assert module.ln_pair._backend != KernelBackend.PYTORCH
        shapes = [(1, 32, 64, 64)]
        if kind == "pwa":
            shapes.append((1, 64, 64, 128))
    else:
        cond = DiffusionConditioning(
            SharedConfig(implementation=ImplementationType.MINIWORLD_ENGINE),
            DiffusionConditioning.Config(),
        )
        module = (
            cond.pair_transitions[0]
            if kind == "pair_cond"
            else cond.single_transitions[0]
        )
        # DiffusionModel keeps the trainable diffusion module in FP32.
        module.float()
        assert module._backend != KernelBackend.PYTORCH
        shapes = [(1, 64, 64, 128)] if kind == "pair_cond" else [(1, 1, 64, 384)]
    module = module.cuda().train(training)
    parameter_dtypes = [p.dtype for p in module.parameters()]
    # Exercise the branch and its gradients; production output projections start at zero.
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "to_out" in name or "squeeze" in name:
                p.normal_(std=0.02)
    reference = copy.deepcopy(module)
    for child in reference.modules():
        if hasattr(child, "_backend"):
            child._backend = KernelBackend.PYTORCH
    inputs = [
        torch.randn(s, device="cuda", dtype=torch.bfloat16, requires_grad=training)
        for s in shapes
    ]
    refs = [x.detach().clone().requires_grad_(training) for x in inputs]
    with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.bfloat16):
        actual, expected = module(*inputs), reference(*refs)
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 0.03, error
    if training:
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        pairs = [
            (f"input_{i}", x.grad, r.grad) for i, (x, r) in enumerate(zip(inputs, refs))
        ]
        pairs += [
            (name, p.grad, r.grad)
            for (name, p), r in zip(module.named_parameters(), reference.parameters())
        ]
        for name, actual_grad, expected_grad in pairs:
            assert actual_grad is not None and expected_grad is not None
            assert torch.isfinite(actual_grad).all()
            if kind == "pwa" and name == "ln_pair.bias":
                # This offset shifts every key logit equally, so its exact
                # derivative is zero. BF16 cancellation noise has no meaningful
                # relative error; bound BOTH paths against the non-null LN scale
                # gradient at BF16 precision instead.
                scale = reference.ln_pair.weight.grad.float().norm()
                bound = torch.finfo(torch.bfloat16).eps * scale
                assert actual_grad.float().norm() < bound
                assert expected_grad.float().norm() < bound
                continue
            error = (
                actual_grad.float() - expected_grad.float()
            ).norm() / expected_grad.float().norm().clamp_min(1e-12)
            assert error < 0.08, (name, error, expected_grad.float().norm())
    assert [p.dtype for p in module.parameters()] == parameter_dtypes
