"""CPU regressions for v2 data/loss consistency; no live training or GPU needed."""
from types import SimpleNamespace as NS

import pytest
import torch

from miniworld.loss.confidence import per_atom_lddt, token_frames
from team_gm.diffusion import EDMScheduler, EuclideanDiffuser


def test_plddt_uses_each_batch_ground_truth_and_resolved_mask():
    gt = torch.tensor([[[0., 0, 0], [1, 0, 0], [0, 0, 0]],
                       [[0., 0, 0], [8, 0, 0], [0, 0, 0]]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    pred = gt.clone()
    pred[:, 2] = 1000  # unresolved atoms cannot change a target
    scores, valid = per_atom_lddt(pred, gt, mask)
    assert torch.equal(valid, mask)
    torch.testing.assert_close(scores[:, :2], torch.ones(2, 2))
    pred[1, 1, 0] = 1
    scores, _ = per_atom_lddt(pred, gt, mask)
    assert scores[0, 0] == 1 and scores[1, 0] == 0


def test_frames_use_batch_specific_validity():
    xyz = torch.tensor([[[[0., 1, 0], [0, 0, 0], [1, 0, 0]]]]).expand(2, -1, -1, -1)
    frames = token_frames(xyz, torch.tensor([[True], [False]]))
    assert frames[2].tolist() == [[True], [False]]


def diffuser():
    scheduler = EDMScheduler(EDMScheduler.EDMSchedulerConfig())
    return EuclideanDiffuser(EuclideanDiffuser.EuclideanConfig(), scheduler)


@pytest.mark.parametrize('padding', [0, 12])
def test_edm_loss_and_grad_are_padding_invariant(padding):
    torch.manual_seed(17)
    d = diffuser()
    gt = torch.randn(2, 6, 3)
    inp = torch.randn_like(gt)
    update = torch.randn_like(gt).requires_grad_()
    sigma = torch.ones(2, 1, 1)
    mask = torch.ones(2, 6, dtype=torch.bool)
    weights = torch.tensor([[1., 2, 1, 3, 1, 1]]).expand(2, -1)
    base = d.cal_loss(gt, inp, update, sigma, mask, weights)
    grad, = torch.autograd.grad(base, update)
    def pad(t):
        return torch.nn.functional.pad(t, (0, 0, 0, padding))
    padded_update = pad(update.detach()).requires_grad_()
    padded = d.cal_loss(pad(gt), pad(inp), padded_update, sigma,
                        torch.nn.functional.pad(mask, (0, padding)),
                        torch.nn.functional.pad(weights, (0, padding), value=1))
    padded_grad, = torch.autograd.grad(padded, padded_update)
    torch.testing.assert_close(base, padded)
    torch.testing.assert_close(grad, padded_grad[:, :6])
    assert not padded_grad[:, 6:].count_nonzero()


def test_edm_empty_example_has_zero_loss_and_grad():
    d = diffuser()
    x = torch.zeros(1, 6, 3)
    update = torch.randn_like(x).requires_grad_()
    loss = d.cal_loss(x, x, update, torch.ones(1, 1, 1), torch.zeros(1, 6, dtype=torch.bool))
    loss.backward()
    assert loss == 0 and not update.grad.count_nonzero()


def test_best_of_n_uses_second_sample():
    from miniworld.models.diffusion.client import Client
    gt = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    batch = NS(structure=NS(atom_pos=gt[None], atom_pos_mask=torch.ones(1, 4, dtype=torch.bool)))
    output = NS(atom_pos_pred=torch.stack([gt * 3, gt]))
    batch.to = lambda **kwargs: batch
    result = Client.test_inference_quality(NS(device='cpu'), batch, output)
    assert result['best_rmsd'] < 1e-5
    assert result['best_lddt'] == pytest.approx(1.)


def test_empty_template_padding_matches_training_convention():
    from miniworld.data.pipeline import ProteinTemplate
    from miniworld.data.features.convert import to_template_features
    import numpy as np
    empty = ProteinTemplate.empty(3).padded(n_templates=4)
    features = to_template_features(empty, np.arange(3))
    assert features.mask.shape[-1] == 4
    assert not features.mask.any() and not features.cb_mask.any()
    assert features.res_type.unique().numel() == 1


def test_template_masks_geometry_but_preserves_interchain_query():
    from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder
    from miniworld.data.features import TemplateFeatures
    from torch import nn
    class IdentityPair(nn.Module):
        def forward(self, x, mask):
            return x
    model = AF3TemplateEmbedder(4, num_channels=4, n_block=0, implementation='PYTORCH')
    model.template_pairformer = IdentityPair()
    model.ln_query = nn.Identity()
    model.ln_out = nn.Identity()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.proj_query.weight.copy_(torch.eye(4))
        model.proj_out.weight.copy_(torch.eye(4))
    template = TemplateFeatures(
        mask=torch.tensor([[True, False, False, False]]),
        ids=torch.zeros(1, 4, 2, dtype=torch.long),
        res_type=torch.zeros(1, 4, 2, dtype=torch.long),
        cb_xyz=torch.full((1, 4, 2, 3), float('nan')),
        cb_mask=torch.zeros(1, 4, 2, dtype=torch.bool),
        bb_xyz=torch.full((1, 4, 2, 3, 3), float('nan')),
        bb_mask=torch.zeros(1, 4, 2, dtype=torch.bool))
    pair = torch.ones(1, 2, 2, 4)
    out = model(pair, template, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(out, pair)
    template.mask.zero_()
    assert not model(pair, template, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool)).count_nonzero()


def test_inference_has_representatives_without_ground_truth(tmp_path, monkeypatch):
    import numpy as np
    from miniworld.data.inference import build as build_module
    from miniworld.data.inference.ccd import CCDResidue
    from miniworld.data.inference.spec import InferenceSpec
    ala = CCDResidue('ALA', np.array(['N', 'CA', 'C', 'O', 'CB']),
                     np.array(['N', 'C', 'C', 'O', 'C']), np.zeros(5),
                     np.arange(15, dtype=np.float32).reshape(5, 3), np.ones(5, dtype=bool))
    monkeypatch.setattr(build_module, 'CCDLookup', lambda path: {'ALA': ala})
    fasta = tmp_path / 'a.fasta'
    fasta.write_text('>protein|Chain:A|polypeptide(L)\nAA\n')
    spec = InferenceSpec(chain_letters={'0': 'A'}, fasta={'A': fasta}, ccd_db=tmp_path)
    batch = build_module.build_inference_batch(spec)
    assert batch.structure.atom_mask.all()
    assert not batch.structure.atom_pos_mask.any()
    assert batch.structure.atom_is_rep.sum() == 2
    assert batch.structure.token_frame_mask.all()
    assert batch.template.mask.shape == (1, 4)
    assert not batch.template.mask.any()


def test_confidence_targets_batch_two_without_gt_leak_into_head():
    from miniworld.models.confidence.client import Client
    cfg = NS(n_plddt_bins=50, n_pde_bins=64, pde_max=32., dist_min=3., dist_max=30.,
             n_pae_bins=64, pae_max=32.)
    gt = torch.tensor([[[0., 1, 0], [0, 0, 0], [1, 0, 0]],
                       [[0., 2, 0], [0, 0, 0], [2, 0, 0]]])
    observed = torch.tensor([[True, True, True], [True, True, False]])
    structure = NS(token_mask=torch.ones(2, 3, dtype=torch.bool), atom_pos=gt,
                   atom_pos_mask=observed, atom_mask=torch.ones_like(observed),
                   atom_is_rep=torch.ones_like(observed),
                   token_frame_atoms=torch.tensor([[[0, 1, 2]] * 3] * 2),
                   token_frame_mask=torch.ones(2, 3, dtype=torch.bool))
    batch = NS(structure=structure, chain=NS(entity_type=torch.zeros(2, 1, dtype=torch.long)),
               scheme=NS(atom_to_token_idx_map=torch.arange(3).expand(2, -1),
                         atom_to_chain_id=torch.zeros(2, 3, dtype=torch.long)))
    captured = []
    def head(single, pair, distances, *args):
        captured.append(distances.clone())
        return {}
    fake = NS(config=NS(model=NS(confidence=cfg), loss=NS(pae_loss=1., align_symmetry_targets=False)), model=head)
    _, targets, masks = Client._confidence_targets_and_logits(fake, batch, gt.clone(), torch.empty(0), torch.empty(0))
    assert targets['plddt'].shape == (2, 3)
    assert not masks['plddt'][1, 2] and not masks['pde'][1, 2].any()
    observed.fill_(True)
    Client._confidence_targets_and_logits(fake, batch, gt.clone(), torch.empty(0), torch.empty(0))
    torch.testing.assert_close(captured[0], captured[1])


def test_confidence_chain_permutation_and_mask_follow_each_other():
    import numpy as np
    from miniworld.loss.symmetry import align_confidence_targets
    torch.manual_seed(12)
    gt = torch.randn(1, 8, 3)
    gt[:, 4:] += torch.tensor([8., 3., 2.])
    order = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[0, 6] = False
    batch = NS(structure=NS(atom_pos=gt, atom_pos_mask=mask, atom_mask=torch.ones_like(mask)),
               atom_ids=[np.array(['N', 'CA', 'C', 'O'] * 2)], chem_comp_ids=[np.array(['ALA'] * 2)],
               reference=NS(space_uid=torch.tensor([[0] * 4 + [1] * 4])),
               scheme=NS(atom_to_chain_id=torch.tensor([[0] * 4 + [1] * 4]),
                         atom_to_token_idx_map=torch.tensor([[0] * 4 + [1] * 4]),
                         token_residue_idx=torch.tensor([[0, 0]]),
                         token_entity_id=torch.tensor([[0, 0]])))
    coords, masks = align_confidence_targets(batch, gt[:, order])
    torch.testing.assert_close(coords, gt[:, order])
    assert torch.equal(masks, mask[:, order])
    assert torch.equal(batch.structure.atom_pos_mask, mask)
    # Different entities must not be swapped.
    batch.scheme.token_entity_id[0, 1] = 1
    coords, _ = align_confidence_targets(batch, gt[:, order])
    torch.testing.assert_close(coords, gt)


def test_confidence_residue_symmetry_swap():
    import numpy as np
    from miniworld.loss.symmetry import align_confidence_targets
    torch.manual_seed(71)
    gt = torch.randn(1, 8, 3)
    order = torch.tensor([0, 1, 2, 3, 4, 5, 7, 6])
    mask = torch.ones(1, 8, dtype=torch.bool)
    batch = NS(structure=NS(atom_pos=gt, atom_pos_mask=mask, atom_mask=mask),
               atom_ids=[np.array(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'])],
               chem_comp_ids=[np.array(['ASP'])], reference=NS(space_uid=torch.zeros(1, 8, dtype=torch.long)),
               scheme=NS(atom_to_chain_id=torch.zeros(1, 8, dtype=torch.long),
                         atom_to_token_idx_map=torch.zeros(1, 8, dtype=torch.long),
                         token_residue_idx=torch.zeros(1, 1, dtype=torch.long),
                         token_entity_id=torch.zeros(1, 1, dtype=torch.long)))
    coords, _ = align_confidence_targets(batch, gt[:, order])
    torch.testing.assert_close(coords, gt[:, order])
