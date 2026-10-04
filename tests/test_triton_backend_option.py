"""An application Triton policy must not escape into native H100 engine kernels."""

from pathlib import Path

import pytest
import torch
from miniworld_engine import settings, kernels
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.exceptions import ImplementationType as EngineImpl
from miniworld.training.engine_backend import configure_engine_backend


@pytest.fixture(autouse=True)
def restore_policy():
    previous = settings.current()
    yield
    settings.configure(**vars(previous))


def test_standalone_layernorm_cannot_select_cuda():
    from miniworld_engine.kernels.layernorm.compile_native import _resolve_bwd_path

    configure_engine_backend("triton")
    settings.configure(layernorm_bwd_path="cuda")
    # No CUDA tensors needed: policy precedes hardware lookup, cache and native override.
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    w = torch.ones(128, dtype=x.dtype)
    stat = torch.zeros(2)
    assert _resolve_bwd_path(2, 128, x, x, w, stat, stat) == "atomic"
    assert _resolve_bwd_path(2, 512, x, x, w, stat, stat) == "persistent"


@pytest.mark.parametrize(
    "phase", ["phase1a_distogram_v110", "phase2a_diffusion_v102", "phase3a_confidence"]
)
def test_training_config_override(phase):
    from hydra import compose, initialize_config_dir
    from miniworld.models.distogram_only.client import Client as Distogram
    from miniworld.models.diffusion.client import Client as Diffusion
    from miniworld.models.confidence.client import Client as Confidence

    client = {"1": Distogram, "2": Diffusion, "3": Confidence}[phase[5]]
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        c = compose(config_name=phase, overrides=["train.engine_backend=triton"])
        assert (
            client.TrainConfig.model_validate(dict(c.train)).engine_backend == "triton"
        )
        assert compose(config_name=phase).train.engine_backend == "auto"
    with pytest.raises(ValueError):
        client.TrainConfig(engine_backend="typo")


def test_mini_model_all_engine_modules_and_checkpoint_layout():
    from hydra import compose, initialize_config_dir
    from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
    from miniworld_engine.modules.transition.module import Transition

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        cfg = compose(config_name="phase1a_distogram_v110")
        cfg.model.trunk.pairformer.n_block = 1
        cfg.model.trunk.msa_module.n_block = 1
        cfg.model.trunk.template_embedder.n_block = 1
        cfg.model.input_feat_embbeder.n_block = 1
        config = MiniSWAModel.Config.model_validate(cfg.model)
    configure_engine_backend("auto")
    before = MiniSWAModel(config)
    configure_engine_backend("triton")
    after = MiniSWAModel(config)
    assert {k: tuple(v.shape) for k, v in before.state_dict().items()} == {
        k: tuple(v.shape) for k, v in after.state_dict().items()
    }
    modules = [m for m in after.modules() if hasattr(m, "_backend")]
    # The policy is about where the engine's own choice lands. On B200 team-gm builds its norm / pointwise
    # modules as PyTorch ops on purpose (team_gm.modules._engine_impl.torch_pointwise); those stay PyTorch.
    engine_chosen = [m for m in modules if getattr(m, "implementation", None) != EngineImpl.PYTORCH]
    assert engine_chosen and all(m._backend.value == "triton" for m in engine_chosen)
    assert all(m._backend.value in ("triton", "pytorch") for m in modules)
    assert any(isinstance(m, Transition) for m in modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_standalone_layernorm_blocks_native_override(monkeypatch):
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda

    configure_engine_backend("triton")
    settings.configure(layernorm_bwd_path="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("Native LayerNorm entered under Triton policy")

    monkeypatch.setattr(ln_cuda, "layer_norm_bwd_cuda", forbidden)
    for width in (128, 512):
        x = torch.randn(
            1, 128, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(width, device="cuda", dtype=x.dtype, requires_grad=True)
        b = torch.randn_like(w, requires_grad=True)
        dy = torch.randn_like(x)
        actual = kernels.layernorm_kernel(x, w, b)
        expected = torch.nn.functional.layer_norm(x, (width,), w, b)
        actual_grads = torch.autograd.grad(actual, (x, w, b), dy)
        expected_grads = torch.autograd.grad(expected, (x, w, b), dy)
        for a, e in zip((actual, *actual_grads), (expected, *expected_grads)):
            assert torch.isfinite(a).all()
            assert (a.float() - e.float()).norm() / e.float().norm().clamp_min(
                1e-8
            ) < 0.02
