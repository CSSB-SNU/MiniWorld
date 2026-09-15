"""Regression coverage for the engine fixes used by the phase-2 audit."""

import copy
import zlib

import pytest
import torch


def test_shape_key_checksum_keeps_cache_keys_and_fullgraph():
    from miniworld_engine.autotune.shape_key import pack

    def f(x):
        return x + pack(128, H=4, HEAD_DIM=32) - pack(128, D=64)

    expected = ((128 * 4096 + 4) * 4096 + 32) * 4096
    expected += zlib.crc32(b"H,HEAD_DIM") & 4095
    assert pack(128, H=4, HEAD_DIM=32) == expected
    compiled = torch.compile(f, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(torch.zeros(2, dtype=torch.float64)),
        f(torch.zeros(2, dtype=torch.float64)),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width,cond_width", [(128, 128), (768, 384)])
@pytest.mark.parametrize("training", [False, True])
def test_conditioned_transition_amp_master_weights(width, cond_width, training):
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    from miniworld_engine.modules.dispatch import KernelBackend

    torch.manual_seed(10)
    fused = ConditionedTransition(width, cond_width, implementation="miniworld").cuda()
    with torch.no_grad():
        fused.squeeze.weight.normal_(std=0.02)
        fused.ada_ln_in.to_scale.weight.normal_(std=0.02)
        fused.ada_ln_in.to_bias.weight.normal_(std=0.02)
    reference = copy.deepcopy(fused)
    reference._backend = KernelBackend.PYTORCH
    reference.ada_ln_in._backend = KernelBackend.PYTORCH
    fused.train(training)
    reference.train(training)
    x = torch.randn(
        1, 1, 128, width, device="cuda", dtype=torch.bfloat16, requires_grad=training
    )
    cond = torch.randn(1, 1, 128, cond_width, device="cuda", requires_grad=training)
    xr, cr = (
        x.detach().clone().requires_grad_(training),
        cond.detach().clone().requires_grad_(training),
    )
    with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.bfloat16):
        y, yr = fused(x, cond), reference(xr, cr)
    assert y.dtype == yr.dtype
    assert torch.isfinite(y).all()
    rel = (y.float() - yr.float()).norm() / yr.float().norm()
    assert rel < 0.025, rel
    assert all(p.dtype == torch.float32 for p in fused.parameters())
    if training:
        grad = torch.randn_like(y)
        y.backward(grad)
        yr.backward(grad)
        pairs = [(x.grad, xr.grad), (cond.grad, cr.grad)]
        pairs += [
            (p.grad, pr.grad)
            for p, pr in zip(fused.parameters(), reference.parameters())
        ]
        for g, gr in pairs:
            assert g is not None and gr is not None
            assert torch.isfinite(g).all()
            error = (g.float() - gr.float()).norm() / gr.float().norm().clamp_min(1e-12)
            assert error < 0.06, error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_transition_fullgraph_and_replay():
    from miniworld_engine.modules.transition.module import Transition

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper CUDA b2b path")
    torch.manual_seed(9)
    model = (
        Transition(128, implementation="miniworld").cuda().eval().requires_grad_(False)
    )
    with torch.no_grad():
        model.squeeze.weight.normal_(std=0.01)
    x = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    fn = torch.compile(model, fullgraph=True, options={"triton.cudagraphs": True})
    with torch.no_grad():
        eager = model(x)
        for _ in range(4):
            torch.compiler.cudagraph_mark_step_begin()
            out = fn(x).clone()
        torch.testing.assert_close(out, eager, rtol=0.02, atol=0.02)
        x.mul_(0.8)
        expected = model(x)
        torch.compiler.cudagraph_mark_step_begin()
        out = fn(x).clone()
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
