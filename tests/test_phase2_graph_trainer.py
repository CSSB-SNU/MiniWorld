"""The CPU-checkable parts of the phase-2 CUDA graph trainer: the Fabric stand-in and the wiring into the training script.

The graph itself (capture, replay against eager, the optimizer step) needs a GPU: ``scripts/b200/bench_phase2_synthetic.py ...
train.cuda_graph=true`` runs it on synthetic batches and logs the replay-vs-eager check.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trainer():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import phase2_graph_trainer

        yield phase2_graph_trainer
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_graph_fabric_reads_the_torchrun_environment(trainer, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    fabric = trainer.GraphFabric()
    assert (fabric.global_rank, fabric.world_size, fabric.device) == (3, 4, torch.device("cuda", 1))
    assert not fabric.is_global_zero


def test_graph_fabric_is_a_noop_without_a_process_group(trainer, monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    fabric = trainer.GraphFabric()
    assert fabric.is_global_zero and fabric.world_size == 1
    metrics = {"total_loss": 1.5}
    assert fabric.all_reduce(metrics) is metrics
    module = torch.nn.Linear(2, 2)
    assert fabric.setup_optimizers("optimizer") == "optimizer"
    with fabric.no_backward_sync(module, enabled=True):
        pass


def test_the_training_script_selects_the_graph_trainer_with_train_cuda_graph():
    source = (ROOT / "scripts/run_miniworld_diffusion_train.py").read_text()
    assert "cfg.train.cuda_graph" in source and "Phase2GraphTrainer(client, cfg)" in source
    assert os.path.exists(ROOT / "scripts/phase2_graph_trainer.py")
