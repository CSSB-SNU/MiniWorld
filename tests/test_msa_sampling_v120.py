"""v1.2.0 MSA depth policy: AF3-like full-depth draw for PDB, uniform prefix elsewhere."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from miniworld.configs.data import MSAConfig
from miniworld.data.dataloader import preprocess as module
from miniworld.data.dataloader.preprocess import Preprocessor
from miniworld.data.pipeline import MSA, ComplexMSA, sample_msa

BUDGET = 256


def _chain(rng, n_rows, length, seed):
    """One chain's a3m: unique random rows over 20 residue types, query at row 0."""
    r = np.random.default_rng(seed)
    seqs = r.integers(0, 20, size=(n_rows, length), dtype=np.int64)
    return MSA(
        seq_id=f"P{seed:012d}",
        sequences={
            "query_sequence": seqs[0],
            "aligned_sequences": seqs,
            "deletions": np.zeros((n_rows, length), dtype=np.int64),
            "deletion_mean": np.zeros(length, dtype=np.float32),
            "profile": np.zeros((length, 32), dtype=np.float32),
        },
        headers={"species": np.array([f"s{i}" for i in range(n_rows)])},
    )


@pytest.fixture(scope="module")
def complex_msa():
    rng = np.random.default_rng(0)
    deep, shallow = _chain(rng, 3000, 30, 1), _chain(rng, 40, 20, 2)
    return ComplexMSA([deep, shallow], missing_policy="query", pairing_mode="no_pairing")


def _row_index(complex_msa, block):
    """Map each chain-0 block row back to its row in the full stack."""
    lo, hi = complex_msa.chain_col_spans[0]
    lookup = {bytes(row): i for i, row in enumerate(complex_msa.sequence[:, lo:hi])}
    return np.array([lookup[bytes(row)] for row in block[:, lo:hi]])


def test_af3_draws_over_full_depth_and_saturates_budget(complex_msa):
    n = complex_msa.sequence.shape[0]
    assert n == 3000
    depths = np.array([
        sample_msa(complex_msa, BUDGET, np.random.default_rng(s), sample_depth="af3")
        .aligned_sequences.shape[1]
        for s in range(300)
    ])
    assert depths.min() >= 1 and depths.max() == BUDGET
    # k ~ U[1, 3000] saturates the 256 budget with probability 1 - 256/3000 = 0.915
    frac = (depths == BUDGET).mean()
    assert 0.86 < frac < 0.96, frac
    # and the non-saturated draws are spread, not stuck at the top
    assert (depths < BUDGET).sum() >= 8


def test_af3_keeps_query_and_reaches_rows_past_the_budget(complex_msa):
    feats = sample_msa(complex_msa, BUDGET, np.random.default_rng(3), sample_depth="af3")
    block = feats.aligned_sequences[0].numpy()
    assert block.shape[0] == BUDGET
    np.testing.assert_array_equal(block[0], complex_msa.sequence[0])
    idx = _row_index(complex_msa, block)
    assert idx[0] == 0
    assert idx.max() > BUDGET, "af3 must sample rows beyond the budget, not the prefix"
    assert len(set(idx.tolist())) == BUDGET, "rows are drawn without replacement"
    assert np.all(np.diff(idx[1:]) > 0), "kept rows stay in best-first order"


def test_uniform_is_unchanged_prefix_within_budget(complex_msa):
    for s in range(20):
        feats = sample_msa(complex_msa, BUDGET, np.random.default_rng(s), sample_depth="uniform")
        block = feats.aligned_sequences[0].numpy()
        k = block.shape[0]
        assert 1 <= k <= BUDGET
        np.testing.assert_array_equal(_row_index(complex_msa, block), np.arange(k))


def test_af3_on_shallow_alignment_never_exceeds_available_rows():
    rng = np.random.default_rng(0)
    small = ComplexMSA([_chain(rng, 12, 16, 7)], missing_policy="query",
                       pairing_mode="no_pairing")
    for s in range(30):
        d = sample_msa(small, BUDGET, np.random.default_rng(s), sample_depth="af3")
        assert 1 <= d.aligned_sequences.shape[1] <= 12


def test_policy_resolves_per_source():
    cfg = MSAConfig(sample_depth_by_source={"pdb": "af3"})
    assert cfg.policy_for("pdb") == "af3"
    assert cfg.policy_for("protein_monomer") == "uniform"
    assert cfg.policy_for("short_protein_monomer") == "uniform"
    assert cfg.policy_for("disordered_pdb") == "uniform"
    assert cfg.policy_for(None) == "uniform"
    assert MSAConfig().policy_for("pdb") == "uniform"  # older configs: no change


def test_preprocessor_passes_the_source_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(module, "sample_msa",
                        lambda **kw: seen.append(kw["sample_depth"]) or "feats")
    pre = Preprocessor.__new__(Preprocessor)
    pre.msa_config = MSAConfig(max_msa_depth=BUDGET, sample_depth_by_source={"pdb": "af3"})
    rng = np.random.default_rng(0)
    for source in ("pdb", "protein_monomer", "pdb"):
        assert pre._sample_msa(object(), NS(source=source), rng) == "feats"
    assert seen == ["af3", "uniform", "af3"]


def test_v120_configs_compose():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld"
    for name in ("phase1a_distogram_v120", "phase1b_distogram_v120",
                 "phase1a_distogram_medium_v120"):
        with initialize_config_dir(str(root), version_base=None):
            cfg = compose(config_name=name)
        msa = MSAConfig(**cfg.data.msa)
        assert msa.policy_for("pdb") == "af3"
        assert msa.policy_for("protein_monomer") == "uniform"
        assert msa.max_msa_depth == 2048
        assert cfg.data.crop.ab_ag_interface_only is True   # v1.1 inherited
        assert cfg.loss.distogram_interchain_weight == 2.0  # v1.1 inherited
        assert "v1.2.0" in cfg.train.run_dir


def test_msa_indices_stay_row_aligned_when_rows_are_skipped():
    """A homolog identical to the query is dropped from the stack; the per-chain row
    indices must drop the same row, or per-chain subsampling reads the wrong homolog
    (or runs off the end -- the IndexError that killed the first v1.2.0 launch)."""
    rng = np.random.default_rng(0)
    deep = _chain(rng, 500, 24, 11)
    # plant duplicates of the query inside the deep chain, past the shallow chain's depth so
    # the whole stacked row equals the query row -> `skip` removes exactly those two rows
    deep.aligned_sequences[123] = deep.aligned_sequences[0]
    deep.aligned_sequences[321] = deep.aligned_sequences[0]
    shallow = _chain(rng, 30, 16, 12)
    cm = ComplexMSA([deep, shallow], missing_policy="query", pairing_mode="no_pairing")
    n = cm.sequence.shape[0]
    assert n == 500 - 2
    for key, idx in cm.msa_indices.items():
        assert idx.shape[0] == n, (key, idx.shape, n)
    # every row the indices point at must be the homolog actually stored in `sequence`
    lo, hi = cm.chain_col_spans[0]
    rows = np.flatnonzero(np.asarray(cm.msa_indices[0]) != -1)
    np.testing.assert_array_equal(cm.sequence[rows, lo:hi],
                                  deep.aligned_sequences[np.asarray(cm.msa_indices[0])[rows]])
    # and drawing (almost) everything must not run off the end
    for seed in range(5):
        feats = sample_msa(cm, n - 1, np.random.default_rng(seed), sample_depth="af3")
        assert 1 <= feats.aligned_sequences.shape[1] <= n - 1
