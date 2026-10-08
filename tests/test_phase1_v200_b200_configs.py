"""The B200 phase 1 v2 configs: v200 medium schedule and crops, B200 data, the retired interchain x2 and the new chemistry weights."""
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_miniworld_distogram_train import Config  # noqa: E402


def load(name):
    with initialize_config_dir(str(ROOT / "configs/miniworld"), version_base=None):
        return Config.model_validate(compose(config_name=name))


@pytest.mark.parametrize("name,crop,epochs", [
    ("phase1a_distogram_medium_v200_b200", 384, 800),
    ("phase1b_distogram_medium_v200_b200", 768, 1000),
])
def test_v200_medium_b200_phase1(name, crop, epochs):
    cfg = load(name)
    base = load(name.replace("_b200", ""))
    # schedule, crop and model are the v200 medium ones
    assert cfg.data.crop.max_tokens == crop == base.data.crop.max_tokens
    assert cfg.train.num_epoch == epochs == base.train.num_epoch
    assert cfg.model.trunk.pairformer.n_block == 16 and cfg.model.trunk.n_recycle_max == base.model.trunk.n_recycle_max > 1
    assert cfg.model.trunk.diffusion is None                      # the recycling CE head, not distogram diffusion
    assert str(cfg.train.precision) == "bf16-mixed"
    assert cfg.loss.distogram_cb_target
    # the retired interchain x2 and the new objective
    assert base.loss.distogram_interchain_weight == 2.0           # what v200 inherited from v1.1
    assert cfg.loss.distogram_interchain_weight == 1.0
    lc = cfg.loss
    assert (lc.distogram_alpha_dna, lc.distogram_alpha_rna, lc.distogram_alpha_ligand, lc.distogram_ab_ag_weight) == (5.0, 5.0, 10.0, 10.0)
    assert lc.distogram_class_metrics
    # B200 server paths, W&B on, effective batch 256 on 4 GPUs, one shared run directory
    assert cfg.data.train_db.resources_base.startswith("/NHNHOME/") and "b200" in str(cfg.data.train_db.catalog_cache_path)
    assert cfg.train.use_wandb and cfg.train.wandb_project == "MiniWorld"
    assert cfg.train.grad_accum_steps == 64
    # no Triton anywhere: no inductor, and the guard that stops the run on any Triton launch
    assert cfg.train.compile is False and cfg.train.forbid_triton is True
    assert cfg.train.run_dir == load("phase1a_distogram_medium_v200_b200").train.run_dir
    # the data policy of v1.2 / v1.1 stays
    assert cfg.data.crop.ab_ag_interface_only and cfg.data.msa.pairing_mode == "no_pairing"
