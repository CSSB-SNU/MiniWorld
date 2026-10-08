"""Chemistry weights of the distogram CE: AF3 Eq. 4 (DNA / RNA / ligand) on token pairs and the antibody-antigen factor."""
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from miniworld.loss.auxiliary import (
    atom_distogram_target,
    cal_atom_distogram_loss,
    distogram_class_ce,
    distogram_pair_class_masks,
    distogram_pair_weight,
)
from miniworld.models.distogram_only.client import Client, _pair_weight, _pair_weights_enabled

# one token per chain: antibody, protein, DNA, RNA, ligand, second antibody, D-protein, other nucleic acid
ENTITY = torch.tensor([[0, 1, 4, 3, 6, 0, 2, 5]])
ASYM = torch.arange(8)[None]
A, P, D, R, L, A2, Q, N = range(8)


def weight(**kw):
    return distogram_pair_weight(ENTITY, ASYM, alpha_dna=5.0, alpha_rna=5.0, alpha_ligand=10.0, ab_ag_weight=10.0, **kw)[0]


def test_defaults_weigh_every_pair_one():
    w = distogram_pair_weight(ENTITY, ASYM)
    assert w.shape == (1, 8, 8) and bool((w == 1).all())


@pytest.mark.parametrize("i,j,expected", [
    (P, Q, 1.0),            # protein - D-protein: no chemistry weight, not an antibody pair
    (A, P, 10.0),           # antibody - antigen
    (A, Q, 10.0),           # antibody - D-protein antigen
    (A, A2, 1.0),           # antibody - antibody: no factor
    (A, D, 6.0),            # antibody - DNA: AF3 weight only (not an antigen)
    (A, R, 6.0),
    (A, L, 11.0),           # antibody - ligand: AF3 weight only
    (P, D, 6.0),            # 1 + alpha_dna
    (P, R, 6.0),            # 1 + alpha_rna
    (P, N, 6.0),            # other nucleic acid counts as RNA
    (P, L, 11.0),           # 1 + alpha_ligand
    (D, R, 11.0),           # 1 + alpha_dna + alpha_rna
    (D, L, 16.0),           # 1 + alpha_dna + alpha_ligand
    (L, L, 11.0),           # either-token indicator: counted once
])
def test_pair_weight_values(i, j, expected):
    w = weight()
    assert float(w[i, j]) == expected and float(w[j, i]) == expected


def test_weight_is_symmetric_and_class_masks_are_either_token():
    w = weight()
    torch.testing.assert_close(w, w.T)
    cls = {k: v[0] for k, v in distogram_pair_class_masks(ENTITY, ASYM).items()}
    assert bool(cls["ab_ag"][A, P]) and bool(cls["ab_ag"][P, A]) and not bool(cls["ab_ag"][A, A2]) and not bool(cls["ab_ag"][A, L])
    assert bool(cls["ligand"][P, L]) and bool(cls["ligand"][L, L]) and not bool(cls["ligand"][P, D])


def test_same_chain_tokens_share_the_chain_entity_type():
    # two tokens of chain 0 (antibody) and two of chain 1 (protein): every cross pair is antibody-antigen, every pair inside a chain is not
    w = distogram_pair_weight(torch.tensor([[0, 1]]), torch.tensor([[0, 0, 1, 1]]), ab_ag_weight=10.0)[0]
    expected = torch.tensor([[1, 1, 10, 10], [1, 1, 10, 10], [10, 10, 1, 1], [10, 10, 1, 1]], dtype=torch.float)
    torch.testing.assert_close(w, expected)


@pytest.mark.parametrize("kw", [dict(alpha_dna=-1.0), dict(alpha_ligand=float("inf")), dict(ab_ag_weight=float("nan"))])
def test_invalid_weights_are_refused(kw):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        distogram_pair_weight(ENTITY, ASYM, **kw)


def toy_inputs():
    pos = torch.tensor([[[0., 0, 0], [1., 0, 0], [10., 0, 0], [100., 0, 0], [3., 0, 0], [50., 0, 0], [7., 0, 0], [20., 0, 0]]])
    mask = torch.ones(1, 8, dtype=torch.bool)
    return pos, mask, torch.arange(8)[None]


def loss(logits, **kw):
    pos, mask, mapping = toy_inputs()
    return cal_atom_distogram_loss(logits, pos, mask, mapping, rep_atom_mask=torch.ones_like(mask), token_asym_id=ASYM, **kw)


def test_unit_weight_is_the_old_loss_bit_for_bit_and_matches_a_hand_computation():
    torch.manual_seed(1)
    logits = torch.randn(1, 8, 8, 96, requires_grad=True)
    old = loss(logits)
    torch.testing.assert_close(loss(logits, pair_weight=torch.ones(1, 8, 8)), old, rtol=0, atol=0)
    pos, mask, mapping = toy_inputs()
    target, _ = atom_distogram_target(atom_pos=pos, atom_pos_mask=mask, atom_to_token_idx_map=mapping, num_bins=96, token_num=8,
                                      rep_atom_mask=torch.ones_like(mask))
    ce = F.cross_entropy(logits.permute(0, 3, 1, 2), target, reduction="none")[0]
    w = weight()
    tri = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    expected_old = (ce * tri).sum() / tri.sum()
    expected_new = (ce * w * tri).sum() / tri.sum()          # the denominator stays the number of valid pairs
    torch.testing.assert_close(old[0], expected_old)
    torch.testing.assert_close(loss(logits, pair_weight=w[None])[0], expected_new)


def test_gradient_is_exactly_the_pair_weight_times_the_unweighted_gradient():
    torch.manual_seed(2)
    logits = torch.randn(1, 8, 8, 96, requires_grad=True)
    g1, = torch.autograd.grad(loss(logits).sum(), logits)
    w = weight()
    g2, = torch.autograd.grad(loss(logits, pair_weight=w[None]).sum(), logits)
    torch.testing.assert_close(g2, g1 * w[None, :, :, None], rtol=1e-6, atol=1e-7)


def test_pair_weight_multiplies_the_interchain_weight():
    torch.manual_seed(3)
    logits = torch.randn(1, 8, 8, 96)
    w = weight()[None]
    both = loss(logits, interchain_weight=2.0, pair_weight=w)
    # every token is its own chain here, so every pair is interchain: the two factors multiply
    torch.testing.assert_close(both, 2.0 * loss(logits, pair_weight=w))


def test_class_metrics_are_unweighted_and_report_the_pair_shares():
    torch.manual_seed(4)
    logits = torch.randn(1, 8, 8, 96)
    pos, mask, mapping = toy_inputs()
    out = distogram_class_ce(logits, pos, mask, mapping, ENTITY, ASYM, rep_atom_mask=torch.ones_like(mask))
    assert set(out) == {"protein", "na", "ligand", "ab_ag", "frac_protein", "frac_na", "frac_ligand", "frac_ab_ag"}
    total = 8 * 7 // 2
    # ab_ag pairs: antibody tokens {A, A2} x protein tokens {P, Q} = 4; ligand pairs: L with the 7 others
    assert out["frac_ab_ag"] == pytest.approx(4 / total) and out["frac_ligand"] == pytest.approx(7 / total)
    assert all(out[k] > 0 for k in ("protein", "na", "ligand", "ab_ag"))
    # the unweighted CE of a class is a plain mean over its pairs, whatever the loss weights are
    unweighted = loss(logits)
    assert float(unweighted) > 0


def test_loss_config_defaults_leave_the_loss_untouched():
    cfg = Client.LossConfig()
    batch = NS(chain=NS(entity_type=ENTITY), scheme=NS(token_asym_id=ASYM))
    assert not _pair_weights_enabled(cfg) and _pair_weight(cfg, batch) is None
    assert cfg.distogram_interchain_weight == 1.0 and not cfg.distogram_class_metrics
    on = Client.LossConfig(distogram_alpha_dna=5, distogram_alpha_rna=5, distogram_alpha_ligand=10, distogram_ab_ag_weight=10)
    assert _pair_weights_enabled(on)
    torch.testing.assert_close(_pair_weight(on, batch)[0], weight())
