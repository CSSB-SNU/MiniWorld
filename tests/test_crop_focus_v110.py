"""Pair-focus sampling policy, including ordering and missing interfaces."""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from miniworld.configs.data import CropConfig
from miniworld.data.dataloader import preprocess as module
from miniworld.data.dataloader.preprocess import Preprocessor, WrongCroppingError
from miniworld.data.pipeline.utils import NoInterfaceError


def setup_pair(monkeypatch, tags, *, draw=0.25, **overrides):
    chains = {
        name: NS(seq_id=NS(value=np.array([tag + "000001"])), atoms=object())
        for name, tag in zip(["left", "right"], tags, strict=True)
    }
    mol = NS(id="test", chains=NS(select=lambda *, chain_id: chains[chain_id]))
    pre = Preprocessor.__new__(Preprocessor)
    pre.crop_config = CropConfig(**{
        "ab_ag_interface_only": True,
        "prefer_nonprotein_focus": True,
        **overrides,
    })
    rng = NS(random=Mock(return_value=draw), choice=Mock(side_effect=lambda xs: xs[-1]))
    interface = Mock(return_value=NS(atoms=object()))
    monkeypatch.setattr(module, "find_interface_residues", interface)
    return pre, mol, chains, rng, interface


@pytest.mark.parametrize("tags", ["AP", "PA", "AQ", "QA"])
@pytest.mark.parametrize("draw", [0.0, 0.5, 0.99])
def test_ab_ag_always_interface(monkeypatch, tags, draw):
    pre, mol, _, rng, interface = setup_pair(
        monkeypatch, tags, draw=draw, chain_crop_prob=1.0,
    )
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is interface.return_value.atoms
    interface.assert_called_once_with(mol, "left", "right")
    rng.random.assert_not_called()
    rng.choice.assert_not_called()


@pytest.mark.parametrize("protein", "PAQ")
@pytest.mark.parametrize("other", "RDNLBX")
@pytest.mark.parametrize("reverse", [False, True])
def test_chain_branch_always_nonprotein(monkeypatch, protein, other, reverse):
    tags = other + protein if reverse else protein + other
    selected = "left" if reverse else "right"
    pre, mol, chains, rng, interface = setup_pair(monkeypatch, tags)
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is chains[selected].atoms
    rng.choice.assert_called_once_with([selected])
    interface.assert_not_called()


@pytest.mark.parametrize("tags", ["PP", "PQ", "AA", "DR", "LL"])
def test_two_proteins_or_two_nonproteins_keep_uniform_choice(monkeypatch, tags):
    pre, mol, chains, rng, interface = setup_pair(monkeypatch, tags)
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is chains["right"].atoms
    rng.choice.assert_called_once_with(["left", "right"])
    interface.assert_not_called()


@pytest.mark.parametrize("tags", ["PP", "PL", "AR", "DR"])
@pytest.mark.parametrize("draw", [0.5, 0.99])
def test_other_pairs_retain_interface_half(monkeypatch, tags, draw):
    pre, mol, _, rng, interface = setup_pair(monkeypatch, tags, draw=draw)
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is interface.return_value.atoms
    rng.choice.assert_not_called()


def test_ab_ag_without_interface_fails_instead_of_using_chain(monkeypatch):
    pre, mol, _, rng, interface = setup_pair(monkeypatch, "AP")
    interface.side_effect = NoInterfaceError("no contact")
    with pytest.raises(WrongCroppingError, match="Ab-Ag.*no interface"):
        pre._select_focus_atoms(mol, ["left", "right"], rng)
    rng.choice.assert_not_called()


def test_other_missing_interface_fallback_uses_nonprotein(monkeypatch):
    pre, mol, chains, rng, interface = setup_pair(monkeypatch, "LP", draw=0.75)
    interface.side_effect = NoInterfaceError("no contact")
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is chains["left"].atoms
    rng.choice.assert_called_once_with(["left"])


@pytest.mark.parametrize("tags", ["AP", "PL"])
@pytest.mark.parametrize("draw", [0.25, 0.75])
def test_old_configs_keep_uniform_chain_and_fallback(monkeypatch, tags, draw):
    pre, mol, chains, rng, interface = setup_pair(
        monkeypatch, tags, draw=draw,
        ab_ag_interface_only=False, prefer_nonprotein_focus=False,
    )
    interface.side_effect = NoInterfaceError("no contact")
    assert pre._select_focus_atoms(mol, ["left", "right"], rng) is chains["right"].atoms
    rng.choice.assert_called_once_with(["left", "right"])


def test_single_chain_unchanged(monkeypatch):
    pre, mol, chains, rng, interface = setup_pair(monkeypatch, "AP")
    assert pre._select_focus_atoms(mol, ["left"], rng) is chains["left"].atoms
    rng.random.assert_not_called()
    rng.choice.assert_not_called()
    interface.assert_not_called()


def test_phase_configs_enable_v110_policy_only():
    root = Path(__file__).resolve().parents[1]
    assert not CropConfig().ab_ag_interface_only
    assert not CropConfig().prefer_nonprotein_focus
    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        for phase, tokens, atoms in [("a", 384, 4096), ("b", 768, 8192)]:
            for suffix in ["", "_v110"]:
                cfg = compose(config_name=f"phase1{phase}_distogram{suffix}")
                crop = CropConfig.model_validate(dict(cfg.data.crop))
                assert crop.ab_ag_interface_only == bool(suffix)
                assert crop.prefer_nonprotein_focus == bool(suffix)
                assert crop.chain_crop_prob == 0.5
                assert (crop.max_tokens, crop.max_atoms) == (tokens, atoms)
