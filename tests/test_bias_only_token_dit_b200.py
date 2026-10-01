"""On B200 in bf16 the v2.0.0 token DiT runs on the engine's fused bias-only paths (CUDA + cuBLAS).

Training (output and every gradient) and inference against the fp32 module path, no worse than the
bf16 module path.
"""

import pytest
import torch
from miniworld_engine.integrations import bias_only_dit, bias_only_dit_train
from team_gm.modules.exceptions import ImplementationType

from miniworld.modules.bias_only_token_dit import BiasOnlyTokenDiT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="B200 (sm_100)",
)
A, L = 4, 256


def _dit(dtype):
    torch.manual_seed(0)
    dit = BiasOnlyTokenDiT(BiasOnlyTokenDiT.Config(
        n_block=2, implementation=ImplementationType.MINIWORLD_ENGINE)).cuda()
    with torch.no_grad():  # to_out / to_bias start at zero: randomise everything
        for p in dit.parameters():
            p.normal_(std=p.shape[-1] ** -0.5 if p.dim() > 1 else 0.1)
    return dit.to(dtype)


def _inputs(dtype):
    g = torch.Generator(device="cuda").manual_seed(1)
    single = torch.randn(A, 1, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, L - 37:] = False
    return [t.to(dtype).requires_grad_() for t in (single, cond, pair)], mask


def _train(dtype):
    dit = _dit(dtype)
    (single, cond, pair), mask = _inputs(dtype)
    out = dit(single, cond, pair, mask)
    w = torch.randn(out.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "single": single.grad.float(), "cond": cond.grad.float(),
           "pair": pair.grad.float()}
    res.update({n: p.grad.float() for n, p in dit.named_parameters()})
    return res


def _rel(x, ref):
    return float((x.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for mod, name in ((bias_only_dit_train, "block"), (bias_only_dit, "update")):
        orig = getattr(mod, name)
        monkeypatch.setattr(mod, name, (lambda o, n: lambda *a, **k: seen.append(n) or o(*a, **k))(orig, name))
    return seen


def test_training_takes_the_fused_path(calls, monkeypatch):
    truth = _train(torch.float32)
    assert calls == []
    fused = _train(torch.bfloat16)
    assert calls == ["block", "block"]
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_TRAIN", "0")
    module = _train(torch.bfloat16)
    assert calls == ["block", "block"]
    assert set(fused) == set(truth)
    for name, t in truth.items():
        ef, em = _rel(fused[name], t), _rel(module[name], t)
        assert ef < 1.5 * em + 3e-3, (name, ef, em)


def test_inference_takes_the_fused_path(calls, monkeypatch):
    outs = {}
    for key, dtype in (("truth", torch.float32), ("fused", torch.bfloat16), ("module", torch.bfloat16)):
        if key == "module":
            monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT", "0")
        dit = _dit(dtype)
        (single, cond, pair), mask = _inputs(dtype)
        with torch.no_grad():
            outs[key] = dit(single, cond, pair, mask).float()
        if key == "fused":
            assert calls == ["update", "update"]
    assert calls == ["update", "update"]
    ef, em = _rel(outs["fused"], outs["truth"]), _rel(outs["module"], outs["truth"])
    assert ef < 1.5 * em + 3e-3, (ef, em)
