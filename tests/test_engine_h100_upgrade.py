import copy
import pytest
import torch
from miniworld_engine.modules import Transition, TriangleMultiplication
from miniworld_engine.modules.dispatch import KernelBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def hopper_only():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper kernel qualification")


def close(a, b, limit=0.025):
    assert a is not None and b is not None
    assert torch.isfinite(a).all()
    error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
    assert error < limit, float(error)


@pytest.mark.parametrize(
    "width,rows", [(256, 128), (384, 128), (512, 128), (512, 16384)]
)
@pytest.mark.parametrize("amp", [False, True])
def test_transition_training_residual(width, rows, amp):
    torch.manual_seed(181)
    actual = Transition(width, implementation="miniworld").cuda().train()
    actual._backend = KernelBackend.CUTE
    if not amp:
        actual.bfloat16()
    with torch.no_grad():
        actual.squeeze.weight.normal_(std=0.02)
    ref = copy.deepcopy(actual).float()
    for module in ref.modules():
        if hasattr(module, "_backend"):
            module._backend = KernelBackend.PYTORCH
    # FP32 input under autocast also verifies residual dtype promotion.
    x = torch.randn(
        1,
        rows,
        width,
        device="cuda",
        dtype=torch.float32 if amp else torch.bfloat16,
        requires_grad=True,
    )
    xr = x.detach().float().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = actual(x)
    yr = ref(xr)
    assert y.dtype == x.dtype
    close(y, yr)
    dy = torch.randn_like(y)
    y.backward(dy)
    yr.backward(dy.float())
    close(x.grad, xr.grad)
    for (name, p), (ref_name, pr) in zip(
        actual.named_parameters(), ref.named_parameters()
    ):
        assert name == ref_name
        close(p.grad, pr.grad)


def test_trimul_compiled_graph_rng_and_mask():
    torch.manual_seed(923)
    model = (
        TriangleMultiplication(128, implementation="miniworld", p_drop=0.25)
        .cuda()
        .bfloat16()
        .train()
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.08)
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    model = torch.compile(model, fullgraph=True)
    # Keep training dropout on, but capture only forward here; backward is tested separately.
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(x, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(x, mask)
        graph.replay()
        first = output.clone()
        graph.replay()
        second = output.clone()
        assert not torch.equal(first, second), "dropout RNG froze during graph replay"
        mask.zero_()
        graph.replay()
        torch.testing.assert_close(output, x, rtol=0, atol=0)


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("rows", [128, 4096])
def test_m2_requalification(width, rows):
    from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
        layernorm_linear_cute_fused,
    )

    torch.manual_seed(817)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(width, width, device="cuda", dtype=torch.bfloat16) * width**-0.5
    gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn_like(gamma)
    bias = torch.randn_like(gamma)
    for source in (x, x.T.contiguous().T):
        actual = layernorm_linear_cute_fused(source, gamma, beta, w, bias, 0.03)
        ref = torch.nn.functional.linear(
            torch.nn.functional.layer_norm(
                source.float(), (width,), gamma.float(), beta.float(), 0.03
            ),
            w.float(),
            bias.float(),
        )
        close(actual, ref)


@pytest.mark.parametrize("amplitude", [1.0, 0.001])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
def test_m2_restored_dispatch_graph(amplitude, norm_dtype):
    from miniworld_engine.kernels.layernorm_linear.cute import layernorm_linear

    torch.manual_seed(385)
    x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16) * amplitude
    w = torch.randn(128, 128, device="cuda", dtype=x.dtype) * 0.05
    gamma = torch.randn(128, device="cuda", dtype=norm_dtype)
    beta = torch.randn_like(gamma)
    bias = torch.randn(128, device="cuda", dtype=x.dtype)
    fn = torch.compile(
        lambda a: layernorm_linear(a, gamma, beta, w, bias, 0.001), fullgraph=True
    )
    reference = lambda a: torch.nn.functional.linear(
        torch.nn.functional.layer_norm(
            a.float(), (128,), gamma.float(), beta.float(), 0.001
        ),
        w.float(),
        bias.float(),
    )
    with torch.no_grad():
        close(fn(x), reference(x))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = fn(x)
        x.normal_().mul_(amplitude)
        graph.replay()
        close(y, reference(x))


@pytest.mark.parametrize("save", [False, True])
def test_masked_front_partial_tile_and_preactivation(save):
    from miniworld_engine.kernels.trimul_inproj.cute.masked_front import masked_front

    torch.manual_seed(838)
    a = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 512, device="cuda", dtype=a.dtype) * 0.08
    mask = torch.ones(192, 2, device="cuda", dtype=torch.bool)[:, 0]
    mask[::3] = False
    actual, preact = masked_front(a, b, mask, save)
    projected = a.float() @ b.float()
    ref = torch.sigmoid(projected[:, ::2]) * projected[:, 1::2] * mask[:, None]
    close(actual, ref)
    assert (actual[~mask] == 0).all()
    if save:
        close(preact, projected)
        assert (preact[~mask] != 0).any()
    else:
        assert preact.numel() == 0


@pytest.mark.parametrize("save,projected", [(False, 256), (True, 512)])
def test_masked_front_cache_key_normalizes_masks(monkeypatch, save, projected):
    from miniworld_engine.autotune import cute_config
    from miniworld_engine.kernels.trimul_inproj.cute.masked_front import masked_front

    keys = []
    original = cute_config.resolve_config

    def observe(op, candidates, **kwargs):
        keys.append(kwargs["bucket"])
        return original(op, candidates, **kwargs)

    monkeypatch.setattr(cute_config, "resolve_config", observe)
    a = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, projected, device="cuda", dtype=a.dtype) * 0.08
    mask = torch.ones(192, device="cuda", dtype=torch.bool)
    mask[::3] = False
    variants = [
        mask,
        mask.reshape(1, 12, 16),
        mask.reshape(1, 1, 12, 16),
        mask.bfloat16(),
    ]
    variants.append(torch.stack([mask, mask], dim=-1)[:, 0])
    expected = None
    for variant in variants:
        output, preact = masked_front(a, b, variant, save)
        if expected is None:
            expected = (output, preact)
        else:
            torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
            torch.testing.assert_close(preact, expected[1], rtol=0, atol=0)
    assert len(keys) == len(variants)
    assert len(set(keys)) == 1, "Equivalent launch masks must reuse one tuning entry"


def test_squeeze_residual_compiled_graph_keeps_inputs():
    from miniworld_engine.kernels.transition.cute.squeeze_residual import (
        squeeze_residual,
    )

    torch.manual_seed(485)
    expand = torch.randn(16384, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, 2048, device="cuda", dtype=expand.dtype) * 0.02
    residual = torch.randn(16384, 512, device="cuda", dtype=expand.dtype)
    original = residual.clone()
    fn = torch.compile(squeeze_residual, fullgraph=True)
    with torch.no_grad():
        close(
            fn(expand, weight, residual),
            expand.float() @ weight.float().T + residual.float(),
        )
        torch.testing.assert_close(residual, original, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(expand, weight, residual)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(expand, weight, residual)
        residual.normal_()
        graph.replay()
        close(output, expand.float() @ weight.float().T + residual.float())


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "bidir"])
@pytest.mark.parametrize("training", [False, True])
def test_cold_compiled_trimul_inference(kind, training, monkeypatch):
    from miniworld_engine.modules.triangle_multiplication import (
        BidirectionalTriangleMultiplication,
    )
    from miniworld_engine.modules.triangle_multiplication import module as shared

    torch._dynamo.reset()
    monkeypatch.setattr(shared, "_CUTE_FNS", None)
    torch.manual_seed(689)
    cls = (
        BidirectionalTriangleMultiplication
        if kind == "bidir"
        else TriangleMultiplication
    )
    options = {} if kind == "bidir" else {"outgoing": kind == "outgoing"}
    actual = (
        cls(128, implementation="miniworld", p_drop=0, **options)
        .cuda()
        .bfloat16()
        .train(training)
    )
    with torch.no_grad():
        for module in actual.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.08)
    ref = copy.deepcopy(actual).float()
    for module in ref.modules():
        if hasattr(module, "_backend"):
            module._backend = KernelBackend.PYTORCH
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    actual.compile(fullgraph=True, options={"triton.cudagraphs": False})
    x.requires_grad_(training)
    xr = x.detach().float().requires_grad_(training)
    with torch.set_grad_enabled(training):
        y, yr = actual(x, mask), ref(xr, mask)
        close(y, yr)
        if training:
            dy = torch.randn_like(y)
            y.backward(dy)
            yr.backward(dy.float())
            close(x.grad, xr.grad)
            for (name, p), (ref_name, pr) in zip(
                actual.named_parameters(), ref.named_parameters()
            ):
                assert name == ref_name
                close(p.grad, pr.grad)
