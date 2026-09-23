"""Engine 2 integration must retain Adam values when parameter strides change."""
import os
import pytest
import torch
from miniworld.training.engine_backend import align_engine_optimizer_state, configure_fused_msa_train


def test_adam_resume_layout_preserves_values():
    pytest.importorskip('miniworld_engine.integrations.optimizer')
    p=torch.nn.Parameter(torch.randn(8,4))
    old=torch.optim.Adam([p]);p.square().sum().backward();old.step()
    saved=old.state_dict()
    q=torch.nn.Parameter(p.detach().t().contiguous().t())
    new=torch.optim.Adam([q]);new.load_state_dict(saved)
    before={k:v.clone() for k,v in new.state[q].items()}
    align_engine_optimizer_state(new)
    for k,v in new.state[q].items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        if v.shape==q.shape:assert v.stride()==q.stride()
    assert align_engine_optimizer_state(new)==0
    q.square().sum().backward();new.step()
    assert torch.isfinite(q).all()


def test_msa_disabled_is_explicit(monkeypatch):
    for key in ('MINIWORLD_PWA_TRAIN','MINIWORLD_OPM_TRAIN'):
        monkeypatch.delenv(key,raising=False)
    configure_fused_msa_train(False)
    assert os.environ['MINIWORLD_PWA_TRAIN']=='0'
    assert os.environ['MINIWORLD_OPM_TRAIN']=='0'
    configure_fused_msa_train(True)
    assert os.environ['MINIWORLD_PWA_TRAIN']=='1'
    assert os.environ['MINIWORLD_OPM_TRAIN']=='1'
