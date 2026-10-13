"""Engine 2 integration must retain Adam values when parameter strides change."""

import pytest
import torch
from miniworld_engine import settings

from miniworld.training.engine_backend import (
    align_engine_optimizer_state,
    configure_fused_msa_train,
)


def test_adam_resume_layout_preserves_values():
    pytest.importorskip("miniworld_engine.integrations.optimizer")
    p = torch.nn.Parameter(torch.randn(8, 4))
    old = torch.optim.Adam([p])
    p.square().sum().backward()
    old.step()
    saved = old.state_dict()
    q = torch.nn.Parameter(p.detach().t().contiguous().t())
    new = torch.optim.Adam([q])
    new.load_state_dict(saved)
    before = {k: v.clone() for k, v in new.state[q].items()}
    align_engine_optimizer_state(new)
    for k, v in new.state[q].items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        if v.shape == q.shape:
            assert v.stride() == q.stride()
    assert align_engine_optimizer_state(new) == 0
    q.square().sum().backward()
    new.step()
    assert torch.isfinite(q).all()


def test_msa_disabled_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "_ACTIVE", settings.current())  # restored at teardown
    configure_fused_msa_train(False)
    assert settings.current().pwa_train is False
    assert settings.current().opm_train is False
    configure_fused_msa_train(True)
    assert settings.current().pwa_train is True
    assert settings.current().opm_train is True
