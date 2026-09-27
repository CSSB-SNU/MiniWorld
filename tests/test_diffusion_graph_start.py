"""Fresh graph dispatch and checkpoint/log identity regression checks."""
import importlib
import copy
from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('random_recycle_graph_trainer')


@pytest.mark.parametrize('trainer', ['cudagraph', 'auto', 'random_cudagraph'])
def test_cli_fresh_diffusion_goes_directly_to_graph(graph, monkeypatch, trainer, tmp_path):
    entry = importlib.import_module('run_miniworld_distogram_train')
    monkeypatch.delenv('MW_RESUME_RUN_SUBDIR', raising=False)
    calls = []
    monkeypatch.setattr(graph, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(entry.train, [
        '--config', str(ROOT / 'configs/miniworld/distogram_diffusion_medium_v130_bioai.yaml'),
        f'train.force_trainer={trainer}', f'train.run_dir={tmp_path}',
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert len(calls) == 1
    cfg, checkpoint, directory = calls[0][0]
    assert checkpoint is None and directory is None
    assert cfg.model.trunk.n_recycle_max == 1 and cfg.train.compile
    assert cfg.loss.distogram_interchain_weight == 2.0


def test_directory_fresh_and_resume_do_not_fork_run(graph, tmp_path):
    cfg = SimpleNamespace(train=SimpleNamespace(run_dir=str(tmp_path)))
    directory = graph.resolve_run_directory(cfg, None, None, {})
    assert directory.is_relative_to(tmp_path)
    state = {'run_dir': str(directory)}
    assert graph.resolve_run_directory(cfg, tmp_path / 'copy.pt', None, state) == directory
    (tmp_path / 'latest_run.txt').write_text(str(directory))
    with pytest.raises(ValueError, match='--ckpt'):
        graph.resolve_run_directory(cfg, None, None, {})


def test_wandb_identity_survives_checkpoint_resume(graph, tmp_path):
    run = tmp_path / 'run'
    run_id = graph.prepare_wandb_id(tmp_path, run, {}, True, lambda: 'test-id')
    def unexpected_generation():
        raise AssertionError('Resume generated a new W&B identity')
    assert graph.prepare_wandb_id(tmp_path, run, {'wandb_run_id': run_id}, False,
                                 unexpected_generation) == run_id
    with pytest.raises(ValueError, match='identity'):
        graph.prepare_wandb_id(tmp_path, run, {}, True, unexpected_generation)
    with pytest.raises(ValueError, match='identity'):
        graph.prepare_wandb_id(tmp_path, run, {'wandb_run_id': 'different'}, False,
                              unexpected_generation)


def test_atomic_checkpoint_keeps_previous_state_on_save_failure(graph, monkeypatch, tmp_path):
    state = {'epoch': 1, 'model_state_dict': {'w': torch.tensor([3.])},
             'optimizer_state_dict': {'state': {0: {'step': torch.tensor(2.)}}},
             'ema_state_dict': {'w': torch.tensor([2.])}, 'wandb_run_id': 'test-id'}
    graph.save_checkpoint(tmp_path, state)
    path = tmp_path / 'checkpoints/last.pt'
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('interrupted write')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        graph.save_checkpoint(tmp_path, dict(state, epoch=2))
    assert path.read_bytes() == original
    loaded = torch.load(path, weights_only=False)
    assert loaded['epoch'] == 1 and loaded['wandb_run_id'] == 'test-id'
    torch.testing.assert_close(loaded['ema_state_dict']['w'], state['ema_state_dict']['w'])


def test_rng_roundtrip_restores_cpu_and_cuda_state(graph, monkeypatch):
    gpu_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    received = []
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: gpu_state.clone())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda s: received.append(s.clone()))
    saved = graph.capture_rng_state()
    expected = (torch.rand(4), np.random.rand(4), random.random())
    graph.restore_rng_state(saved)
    actual = (torch.rand(4), np.random.rand(4), random.random())
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert torch.equal(received[0], gpu_state)


def test_preflight_adam_update_cannot_mutate_checkpoint(graph):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved = copy.deepcopy({'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()})
    graph.restore_training_state(model, optimizer, None, saved)
    optimizer.step()
    assert all(s['step'].item() == 1 for s in saved['optimizer_state_dict']['state'].values())
    graph.restore_training_state(model, optimizer, None, saved)
    assert all(s['step'].item() == 1 for s in optimizer.state.values())
