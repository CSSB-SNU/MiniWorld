"""v1.2.0 per-recycle MSA subsampling (AF3 SI 3.3): row draw and trunk wiring."""

from types import SimpleNamespace as NS

import torch

from miniworld.configs.data import MSAConfig
from miniworld.data.features.features import MSAFeatures
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.modules import msa_util
from miniworld.modules.msa_util import subsample_msa_rows

B, N, L, C = 2, 64, 9, 32


def _pool(n_valid: int, seed: int = 0) -> MSAFeatures:
    g = torch.Generator().manual_seed(seed)
    seqs = torch.randint(0, 20, (B, N, L), generator=g)
    seqs[:, :, 0] = torch.arange(N)[None, :] % 20     # row identity in column 0 ...
    seqs[:, :, 1] = torch.arange(N)[None, :] // 20    # ... and column 1
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :n_valid] = True
    return MSAFeatures(
        aligned_sequences=seqs,
        mask=mask,
        has_deletion=torch.randint(0, 2, (B, N, L), generator=g),
        deletion_value=torch.rand(B, N, L, generator=g),
        profile=torch.rand(B, L, C, generator=g),
        deletion_mean=torch.rand(B, L, generator=g),
    )


def _rows(sub: MSAFeatures) -> torch.Tensor:
    s = sub.aligned_sequences
    return s[:, :, 0] + 20 * s[:, :, 1]   # recover the pool row index of each kept row


def test_fixed_output_depth_query_kept_and_order_preserved():
    pool = _pool(n_valid=N)
    sub = subsample_msa_rows(pool, 16)
    assert sub.aligned_sequences.shape == (B, 16, L)
    assert sub.mask.shape == (B, 16)
    assert sub.has_deletion.shape == (B, 16, L) and sub.deletion_value.shape == (B, 16, L)
    rows = _rows(sub)
    assert torch.all(rows[:, 0] == 0), "query row must always be kept"
    assert torch.all(rows[:, 1:] > rows[:, :-1]), "kept rows stay in best-first order"
    assert torch.equal(sub.profile, pool.profile) and torch.equal(sub.deletion_mean, pool.deletion_mean)


def test_valid_rows_are_exhausted_before_padding():
    pool = _pool(n_valid=20)
    sub = subsample_msa_rows(pool, 16)
    assert torch.all(sub.mask), "with 20 valid rows a 16-row draw must contain no padding"
    assert torch.all(_rows(sub) < 20)
    sub = subsample_msa_rows(pool, 40)
    assert sub.mask.sum(dim=1).tolist() == [20, 20], "all 20 valid rows kept, then padding"
    assert torch.all(_rows(sub)[:, :20] < 20)


def test_each_call_draws_new_rows_and_batches_are_independent():
    pool = _pool(n_valid=N)
    torch.manual_seed(0)
    a, b = subsample_msa_rows(pool, 16), subsample_msa_rows(pool, 16)
    assert not torch.equal(_rows(a), _rows(b)), "consecutive recycles must see different rows"
    assert not torch.equal(_rows(a)[0], _rows(a)[1]), "batch items draw independently"
    # per-row gather is consistent across the three row tensors
    ref = torch.gather(pool.deletion_value, 1, _rows(a).unsqueeze(-1).expand(-1, -1, L))
    assert torch.equal(a.deletion_value, ref)


def test_pool_not_larger_than_request_passes_through():
    pool = _pool(n_valid=N)
    assert subsample_msa_rows(pool, N) is pool
    assert subsample_msa_rows(pool, N + 5) is pool


def _bare_model(n_sub):
    m = MiniSWAModel.__new__(MiniSWAModel)
    m.config = NS(shared=NS(num_res_class=C),
                  trunk=NS(msa_subsample_per_recycle=n_sub))
    return m


def test_trunk_step_embeds_only_the_subset(monkeypatch):
    seen = []
    real = msa_util.init_msa
    def spy(msa, **kw):
        seen.append(msa.aligned_sequences.shape[1])
        return real(msa, **kw)
    monkeypatch.setattr("miniworld.models.distogram_only.model_mini_swa.init_msa", spy)
    m = _bare_model(16)
    pool = _pool(n_valid=N)
    feat1, mask1 = m._msa_for_step(pool, None)
    feat2, mask2 = m._msa_for_step(pool, None)
    assert feat1.shape[:2] == (B, 16) and mask1.shape == (B, 16)
    assert feat1.shape[-1] == C + 2 and feat1.dtype == torch.bfloat16
    assert seen == [16, 16], "the MSA module must be handed the 16-row subset, never the pool"
    assert not torch.equal(feat1, feat2), "two recycles, two different subsets"


def test_legacy_path_passes_embedded_pool_through():
    m = _bare_model(None)
    feat = torch.zeros(B, N, L, C + 2, dtype=torch.bfloat16)
    mask = torch.ones(B, N, dtype=torch.bool)
    out_feat, out_mask = m._msa_for_step(feat, mask)
    assert out_feat is feat and out_mask is mask


def test_legacy_trunk_config_default_is_off():
    cfg = MiniSWAModel.TrunkConfig.model_construct()
    assert cfg.msa_subsample_per_recycle is None


def test_pool_size_resolves_per_source():
    cfg = MSAConfig(max_msa_depth=2048, max_msa_depth_by_source={"pdb": 8192})
    assert cfg.depth_for("pdb") == 8192
    assert cfg.depth_for("protein_monomer") == 2048
    assert cfg.depth_for(None) == 2048
    assert MSAConfig(max_msa_depth=2048).depth_for("pdb") == 2048


def test_v120_configs_wire_both_halves():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld"
    for name in ("phase1a_distogram_v120", "phase1b_distogram_v120",
                 "phase1a_distogram_medium_v120"):
        with initialize_config_dir(str(root), version_base=None):
            cfg = compose(config_name=name)
        msa = MSAConfig(**cfg.data.msa)
        assert cfg.model.trunk.msa_subsample_per_recycle == 1024
        assert msa.depth_for("pdb") == 8192 and msa.policy_for("pdb") == "af3"
        assert msa.depth_for("protein_monomer") == 2048
        assert msa.policy_for("protein_monomer") == "uniform"
        assert cfg.train.bucket_msa_multiple >= max(msa.max_msa_depth_by_source.values())
        assert "v1.2.0" in cfg.train.run_dir
