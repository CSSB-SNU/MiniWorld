"""Phase 2: the frozen trunk's recycle depth in training (``model.train_recycle``), its torch.compile-safe draw, and the bucket
warm-up that compiles every depth the real forward can draw. CPU only: the trunk pieces are stubs."""

import contextlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from miniworld.models.diffusion.model import DiffusionModel

ROOT = Path(__file__).resolve().parents[1]
SCHEME = SimpleNamespace(token_asym_id=None)


def _bare_model(train_recycle, *, seed=0, n_max=4):
    """A DiffusionModel without its parameters: the recycle loop of ``_condition_impl`` on stub trunk pieces. Every stub
    trunk step adds 1 to the pair, so the pair that comes out IS the recycle depth that ran."""
    m = DiffusionModel.__new__(DiffusionModel)
    torch.nn.Module.__init__(m)
    m.config = SimpleNamespace(train_recycle=train_recycle)
    m.n_recycle_max = n_max
    m.rng = np.random.default_rng(seed)
    m._forced_n_recycle = None
    m.freeze_trunk = True
    m.dit_dtype = torch.float32
    m._embed = lambda *_: (
        torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 4),
        None,
        None,
        None,
    )
    m._trunk_step = lambda pair, *_: pair + 1
    m.train()
    return m


def _depth(m):
    _, pair = m._condition_impl(None, None, SCHEME, None, None)
    return int(pair[0, 0, 0, 0])


def test_the_default_policy_is_the_full_depth():
    assert DiffusionModel.Config.model_fields["train_recycle"].default == "max"
    m = _bare_model("max")
    assert {_depth(m) for _ in range(20)} == {4}


def test_random_draws_uniformly_from_the_trunk_rng():
    reference = np.random.default_rng(3)
    expected = [int(reference.integers(1, 5)) for _ in range(200)]
    m = _bare_model("random", seed=3)
    drawn = [_depth(m) for _ in range(200)]
    assert drawn == expected
    assert set(drawn) == {1, 2, 3, 4}


def test_one_draw_per_forward():
    m = _bare_model("random", seed=5)
    _depth(m)
    reference = np.random.default_rng(5)
    reference.integers(1, 5)
    assert m.rng.bit_generator.state == reference.bit_generator.state


def test_eval_runs_the_full_depth_and_a_forced_count_wins():
    m = _bare_model("random", seed=1)
    m.eval()
    assert {_depth(m) for _ in range(20)} == {4}
    m.train()
    m._forced_n_recycle = 2
    assert {_depth(m) for _ in range(20)} == {2}


@pytest.mark.parametrize("name", ["phase2a_diffusion_v200", "phase2b_diffusion_v200"])
def test_v200_configs_train_with_random_recycle_and_without_checkpointing(
    name, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    script = importlib.import_module("run_miniworld_diffusion_train")
    with initialize_config_dir(str(ROOT / "configs/miniworld"), version_base=None):
        cfg = script.Config.model_validate(
            compose(config_name=f"{name}.yaml", overrides=["train.use_wandb=False"])
        )
    assert cfg.model.train_recycle == "random"
    assert cfg.model.diffusion.token_dit.n_checkpoint_segments is None
    assert cfg.model.diffusion.atom_swa.n_checkpoint_segments is None


def _warmup_stubs(monkeypatch, train_recycle):
    script = importlib.import_module("run_miniworld_diffusion_train")
    model = _bare_model(train_recycle, seed=11)
    model.weight = torch.nn.Parameter(torch.zeros(1))
    original_rng = model.rng
    steps = []

    class _Client:
        def __init__(self):
            self.model = model
            self.device = torch.device("cpu")
            self.is_global_zero = False
            self.fabric = SimpleNamespace(
                barrier=lambda: None,
                no_backward_sync=lambda *a, **k: contextlib.nullcontext(),
            )
            self.optimizer = SimpleNamespace(
                state_dict=dict,
                load_state_dict=lambda _: None,
                zero_grad=lambda **_: None,
                step=lambda: None,
            )
            self.logger = SimpleNamespace(info=lambda *a, **k: None)

        def training_step(self, _batch):
            steps.append(
                (
                    model._forced_n_recycle,
                    model._draw_train_recycle() if train_recycle == "random" else None,
                )
            )

    cfg = SimpleNamespace(
        data=SimpleNamespace(
            msa=SimpleNamespace(max_msa_depth=128),
            crop=SimpleNamespace(max_tokens=128, max_atoms=1024),
        ),
        train=SimpleNamespace(
            bucket_msa_multiple=128, bucket_token_multiple=128, bucket_atom_multiple=1024
        ),
        model=SimpleNamespace(shared=SimpleNamespace(num_res_class=32)),
    )
    monkeypatch.setattr(script, "_build_precompile_batch", lambda **_: SimpleNamespace())
    return script, model, original_rng, steps, _Client(), cfg


def test_random_warmup_runs_every_recycle_depth_through_the_real_draw(monkeypatch):
    script, model, original_rng, steps, client, cfg = _warmup_stubs(
        monkeypatch, "random"
    )
    script._warmup_bucket_shapes(client, cfg)
    assert steps == [
        (None, 1),
        (None, 2),
        (None, 3),
        (None, 4),
    ]  # unforced: the compiled graph is the one training draws
    assert model.rng is original_rng and model._forced_n_recycle is None


def test_max_warmup_runs_the_full_depth_once(monkeypatch):
    script, model, original_rng, steps, client, cfg = _warmup_stubs(monkeypatch, "max")
    script._warmup_bucket_shapes(client, cfg)
    assert steps == [(None, None)]
    assert model.rng is original_rng
