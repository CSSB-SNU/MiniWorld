"""The graph trainer carries the distogram chemistry pair weight in its static batch and refreshes it for every batch it loads."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import random_recycle_graph_trainer as G  # noqa: E402
import run_miniworld_distogram_train as R  # noqa: E402

from miniworld.models.distogram_only.client import Client, _pair_weight  # noqa: E402


def synthetic():
    return R._build_precompile_batch(device=torch.device("cpu"), msa_depth=16, n_tokens=8, n_atoms=16, n_templates=1, num_res_class=32)


def chemistry_loss(**kw):
    return Client.LossConfig(distogram_alpha_dna=5, distogram_alpha_rna=5, distogram_alpha_ligand=10, distogram_ab_ag_weight=10, **kw)


def test_pair_weight_is_attached_and_follows_every_loaded_batch():
    static, other = synthetic(), synthetic()
    cfg = chemistry_loss()
    assert G.attach_pair_weight(static, cfg)
    assert static.pair_weight.shape == (1, 8, 8)
    other.chain.entity_type = torch.tensor([[0, 1, 4]])                        # antibody, protein, DNA chains
    other.scheme.token_asym_id = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2]], dtype=static.scheme.token_asym_id.dtype)
    G.copy_static(static, other)
    expected = _pair_weight(cfg, other)
    torch.testing.assert_close(static.pair_weight, expected)
    assert float(static.pair_weight[0, 0, 2]) == 10.0 and float(static.pair_weight[0, 2, 4]) == 6.0 and float(static.pair_weight[0, 0, 1]) == 1.0


def test_nothing_is_attached_when_every_weight_is_one():
    static = synthetic()
    assert not G.attach_pair_weight(static, Client.LossConfig())
    assert not hasattr(static, "pair_weight")
    G.copy_static(static, synthetic())                                          # still loads plain batches


def test_graph_loss_reads_the_buffer():
    static = synthetic()
    G.attach_pair_weight(static, chemistry_loss())
    cfg = type("C", (), {"loss": chemistry_loss(distogram_cb_target=True, distogram_loss=1.0), "model": type("M", (), {"trunk": type("T", (), {"diffusion": None})()})()})()
    loss_fn, kwargs = G.graph_objective(cfg)
    assert kwargs == {}
    static.structure.atom_pos = torch.randn_like(static.structure.atom_pos)
    static.structure.atom_pos_mask = torch.ones_like(static.structure.atom_pos_mask)
    logits = torch.randn(1, 8, 8, 96)
    base = loss_fn(logits, static)
    static.pair_weight.mul_(2.0)
    torch.testing.assert_close(loss_fn(logits, static), 2.0 * base)
