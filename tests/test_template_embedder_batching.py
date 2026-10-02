"""AF3TemplateEmbedder on a B200: the templates as one batch of the engine's native TriMul, the four small projections as one linear map.

bf16 is the training dtype (the model builds the embedder in bf16 and feeds it the bf16 trunk pair). Both are the same function as the
per-template loop / the four projections up to bf16 rounding. fp32 keeps the original projections: the aligned fused GEMM would run on
tensor cores under the "medium" matmul precision the training scripts set, the misaligned original ones do not.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from miniworld.modules import template_embedder_af3 as te
from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder

b200 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="B200 (sm_100)",
)


def _engine_with(monkeypatch, **attrs):
    stub = types.ModuleType("miniworld_engine.integrations.trimul_b200")
    for name, value in attrs.items():
        setattr(stub, name, value)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)


def test_an_engine_without_batched_samples_has_a_limit_of_one(monkeypatch):
    _engine_with(monkeypatch)  # the module exists, ``MAX_BATCH`` does not: the import raises ImportError
    assert te._engine_batch_limit() == 1


def test_the_limit_is_what_the_engine_states(monkeypatch):
    _engine_with(monkeypatch, MAX_BATCH=8)
    assert te._engine_batch_limit() == 8


def test_the_templates_are_not_batched_without_a_b200(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not AF3TemplateEmbedder(d_pair=16)._batch_templates


def _inputs(dtype):
    from tests.test_af3_template import make_inputs

    pair, tmpl, asym, tmask = make_inputs()
    return pair.to(dtype), tmpl, asym, tmask


def _embedder(dtype):
    """A trained-like embedder: the pair stack's output projections start at zero (the stack is the identity then, and a
    batched stack that mixed up its samples would go unnoticed), so every matrix gets weights."""
    from tests.test_af3_template import D_PAIR

    emb = AF3TemplateEmbedder(d_pair=D_PAIR).cuda().eval().to(dtype)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        for p in emb.parameters():
            if p.ndim >= 2:
                p.copy_((torch.randn(p.shape, device=p.device, generator=gen) * p.shape[-1] ** -0.5).to(p.dtype))
    return emb


def _rel(x, ref):
    return float((x.float() - ref.float()).norm() / ref.float().norm())


@b200
def test_bf16_fused_projections_match_the_four_projections():
    emb = _embedder(torch.bfloat16)
    emb._batch_templates = False
    args = _inputs(torch.bfloat16)
    with torch.no_grad():
        emb._fuse_projections = True
        fused = emb(*args)
        emb._fuse_projections = False
        plain = emb(*args)
    assert torch.isfinite(fused).all()
    assert _rel(fused, plain) < 2e-2


@b200
def test_bf16_batched_templates_match_the_loop():
    emb = _embedder(torch.bfloat16)
    args = _inputs(torch.bfloat16)
    n_templates = args[1].mask.shape[1]
    if te._engine_batch_limit() < n_templates:
        pytest.skip("the pinned engine takes one sample per TriMul call")
    with torch.no_grad():
        emb._batch_templates = True
        batched = emb(*args)
        emb._batch_templates = False
        looped = emb(*args)
    assert torch.isfinite(batched).all()
    assert _rel(batched, looped) < 2e-2


def _gradients(emb, args, *, batched, fused):
    emb._batch_templates, emb._fuse_projections = batched, fused
    emb.zero_grad(set_to_none=True)
    out = emb(*args)
    out.backward(torch.ones_like(out))
    return {n: p.grad.detach().clone() for n, p in emb.named_parameters() if p.grad is not None}


def _assert_same_gradients(got, want):
    assert got.keys() == want.keys()
    for name, grad in got.items():
        if want[name].norm() == 0:
            continue
        cosine = F.cosine_similarity(grad.flatten().float(), want[name].flatten().float(), dim=0)
        assert cosine > 0.99, (name, float(cosine))


@b200
def test_bf16_batched_templates_give_the_loops_gradients():
    emb = _embedder(torch.bfloat16)
    args = _inputs(torch.bfloat16)
    if emb._batch_limit < args[1].mask.shape[1]:
        pytest.skip("the pinned engine takes one sample per TriMul call")
    _assert_same_gradients(_gradients(emb, args, batched=True, fused=True), _gradients(emb, args, batched=False, fused=True))


@b200
def test_bf16_fused_projections_give_the_four_projections_gradients():
    emb = _embedder(torch.bfloat16)
    args = _inputs(torch.bfloat16)
    _assert_same_gradients(_gradients(emb, args, batched=False, fused=True), _gradients(emb, args, batched=False, fused=False))


@b200
def test_fp32_keeps_the_four_projections(monkeypatch):
    """fp32 never takes the fused map, whatever the flag says: the output is the independent reference's, exactly."""
    from tests.test_af3_template import ref_loop

    emb = _embedder(torch.float32)
    args = _inputs(torch.float32)
    torch.set_float32_matmul_precision("medium")  # what the training scripts set; the aligned fused GEMM would lose precision here
    try:
        with torch.no_grad():
            out = emb(*args)
            ref = ref_loop(emb, *args)
    finally:
        torch.set_float32_matmul_precision("highest")
    assert (out - ref).abs().max().item() < 1e-4
