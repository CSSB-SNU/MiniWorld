"""The capture-safe stand-ins of ``src/miniworld/training/phase2_graph_safe.py`` against the repo's own host-synchronising versions."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from team_gm.diffusion import EDMScheduler, EuclideanDiffuser
from team_gm.utils.align import weighted_align

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gs():
    return importlib.import_module("miniworld.training.phase2_graph_safe")


def _clouds(n=8, length=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, length, 3, generator=g) * 20
    w = (torch.rand(n, length, generator=g) > 0.1).float()
    noise = 0.5 * torch.randn(n, length, 3, generator=g)
    return x, w, noise, g


def _mse(aligned, y, w):
    return (((aligned - y).pow(2).sum(-1) * w).sum(-1) / w.sum(-1)).mean()


def test_horn_alignment_equals_the_svd_alignment(gs):
    x, w, noise, g = _clouds()
    q = torch.randn(x.shape[0], 4, generator=g)
    rot = gs.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    cases = {
        "rotation + noise": torch.bmm(x, rot.transpose(-1, -2)) + noise,
        "mirror image (the reflection branch)": torch.bmm(
            x * torch.tensor([1.0, 1.0, -1.0]), rot.transpose(-1, -2)
        )
        + noise,
        "unrelated clouds": torch.randn_like(x) * 20,
    }
    for name, y in cases.items():
        reference, got = weighted_align(x, y, w), gs.weighted_align_gs(x, y, w)
        assert _mse(got, y, w) <= _mse(reference, y, w) * (1 + 1e-4) + 1e-6, name
    y = cases["rotation + noise"]
    assert (gs.weighted_align_gs(x, y, w) - weighted_align(x, y, w)).abs().max() < 1e-2


def test_horn_alignment_survives_an_extended_almost_collinear_chain(gs):
    """A 1100 A rod: the top two eigenvalues of Horn's matrix are nearly degenerate and a short iteration does not converge."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 1024, 3, generator=g) * torch.tensor([330.0, 2.0, 10.0])
    q = torch.randn(4, 4, generator=g)
    y = torch.bmm(
        x, gs.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True)).transpose(-1, -2)
    ) + 0.1 * torch.randn_like(x)
    w = torch.ones(4, 1024)
    assert (
        _mse(gs.weighted_align_gs(x, y, w), y, w)
        <= _mse(weighted_align(x.double(), y.double(), w.double()).float(), y, w) * 1.001
    )


def test_the_rotation_sampler_is_a_uniform_rotation(gs):
    q = torch.randn(100_000, 4, generator=torch.Generator().manual_seed(2))
    rot = gs.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    assert (rot @ rot.transpose(-1, -2) - torch.eye(3)).abs().max() < 1e-5
    assert torch.linalg.det(rot).min() > 0.9999
    assert abs(float((rot[:, 0, 0] ** 2).mean()) - 1 / 3) < 0.01


def test_rot_trans_keeps_the_pairwise_distances(gs):
    x = torch.randn(3, 64, 3) * 10
    out = gs.rot_trans_gs(
        SimpleNamespace(config=SimpleNamespace(translation_noise=1.0)), x
    )
    assert out.shape == x.shape
    exact = {
        "compute_mode": "donot_use_mm_for_euclid_dist"
    }  # the matmul form of cdist is not exact near 0
    assert torch.allclose(
        torch.cdist(out, out, **exact), torch.cdist(x, x, **exact), atol=1e-3
    )


def _diffuser():
    return EuclideanDiffuser(
        EuclideanDiffuser.EuclideanConfig(seed=0),
        EDMScheduler(EDMScheduler.EDMSchedulerConfig()),
    )


def test_the_graph_safe_loss_equals_the_repos_loss(gs):
    diffuser = _diffuser()
    length = 256
    g = torch.Generator().manual_seed(3)
    pos = torch.randn(1, length, 3, generator=g) * 15
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[:, -20:] = False
    x0, x_input, x_mask, _, sigma = diffuser.sample(pos, num_augment=4, mask=mask)
    update = torch.randn(x0.shape, generator=g) * 3
    weight = torch.ones(1, length)
    reference = diffuser.cal_loss(
        x0=x0,
        x_input=x_input,
        x_update=update,
        sigma=sigma,
        mask=x_mask,
        atom_weight=weight,
    )
    got = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, weight)
    assert abs(float(reference) - float(got)) < 1e-4 * abs(float(reference))


def _eager_bond_loss(diffuser, x0, x_input, update, sigma, pairs):
    """``DiffusionClient.loss_fn``'s AF3 Eq.5 bond term, verbatim, on a sparse ``[n_bond, 2]`` pair list."""
    x_pred = diffuser.get_x_pred(x_input=x_input, x_update=update, sigma=sigma)
    bi, bj = pairs[:, 0].long(), pairs[:, 1].long()
    d_pred = (x_pred[..., bi, :] - x_pred[..., bj, :]).norm(dim=-1)
    d_gt = (x0[..., bi, :] - x0[..., bj, :]).norm(dim=-1)
    per_augment = (d_pred - d_gt).pow(2).mean(dim=-1)
    w_sigma = diffuser.scheduler.loss_weight(sigma).to(dtype=per_augment.dtype).reshape(per_augment.shape)
    return (w_sigma * per_augment).mean()


def _padded_pairs(pairs, size):
    bond_i, bond_j = torch.zeros(size, dtype=torch.long), torch.zeros(size, dtype=torch.long)
    bond_valid = torch.zeros(size, dtype=torch.bool)
    n = pairs.shape[0]
    bond_i[:n], bond_j[:n], bond_valid[:n] = pairs[:, 0], pairs[:, 1], True
    return bond_i, bond_j, bond_valid


def test_the_graph_safe_bond_loss_equals_the_repos_bond_loss(gs):
    diffuser = _diffuser()
    length = 256
    g = torch.Generator().manual_seed(4)
    pos = torch.randn(1, length, 3, generator=g) * 15
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[:, -20:] = False
    x0, x_input, _, _, sigma = diffuser.sample(pos, num_augment=4, mask=mask)
    update = (torch.randn(x0.shape, generator=g) * 3).requires_grad_()
    pairs = torch.tensor([[3, 4], [10, 40], [100, 101], [7, 200], [55, 56]])

    reference = _eager_bond_loss(diffuser, x0, x_input, update, sigma, pairs)
    (ref_grad,) = torch.autograd.grad(reference, update)
    # the same pairs in a padded list (padding is masked, wherever it points)
    got = gs.bond_loss_gs(diffuser, x0, x_input, update, sigma, *_padded_pairs(pairs, 64))
    (got_grad,) = torch.autograd.grad(got, update)
    assert float(reference) > 0
    assert abs(float(reference) - float(got)) < 1e-5 * abs(float(reference))
    assert (ref_grad - got_grad).abs().max() < 1e-5 * ref_grad.abs().max()


def test_the_graph_safe_bond_loss_is_zero_and_finite_without_bonds(gs):
    diffuser = _diffuser()
    g = torch.Generator().manual_seed(5)
    pos = torch.randn(1, 128, 3, generator=g) * 15
    x0, x_input, _, _, sigma = diffuser.sample(pos, num_augment=4, mask=torch.ones(1, 128, dtype=torch.bool))
    update = (torch.randn(x0.shape, generator=g) * 3).requires_grad_()
    empty = _padded_pairs(torch.zeros(0, 2, dtype=torch.long), 16)
    loss = gs.bond_loss_gs(diffuser, x0, x_input, update, sigma, *empty)
    (grad,) = torch.autograd.grad(loss, update)
    assert float(loss) == 0.0 and torch.isfinite(grad).all() and float(grad.abs().max()) == 0.0


def _loss_inputs(seed=6, length=256, n_aug=4):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(1, length, 3, generator=g) * 15
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[:, -20:] = False
    diffuser = _diffuser()
    x0, x_input, x_mask, _, sigma = diffuser.sample(pos, num_augment=n_aug, mask=mask)
    update = torch.randn(x0.shape, generator=g) * 3
    # AF3 Eq.4: a polymer chain, then a nucleic-acid stretch (w = 6) and a ligand (w = 11)
    weight = torch.ones(1, length)
    weight[:, 120:180] = 6.0
    weight[:, 200:230] = 11.0
    return diffuser, x0, x_input, x_mask, sigma, update, weight


def _weighted_sq_error(diffuser, x0, x_input, x_mask, sigma, update, weight, align_weight):
    """The loss's squared error, aligning with ``align_weight`` (the repo's SVD ``weighted_align``), summed with the Eq.4 weights."""
    sch = diffuser.scheduler
    noisy = x_input / sch.input_scale(sigma)
    x_pred = sch.skip_scale(sigma) * noisy + sch.output_scale(sigma) * update
    mask = x_mask.expand(x_pred.shape[:-1])
    x0_safe, pred_safe = torch.where(mask[..., None], x0, 0.0), torch.where(mask[..., None], x_pred, 0.0)
    aligned = weighted_align(x0_safe, pred_safe, weight=align_weight.expand(mask.shape) * mask)
    sq = torch.where(mask[..., None], (pred_safe - aligned).pow(2), 0.0)
    return (sq.sum(-1) * weight.expand(mask.shape)).sum(-1)  # [A, B], Eq.4-weighted


def test_the_default_alignment_is_still_the_mask_only_one_of_the_repos_loss(gs):
    diffuser, x0, x_input, x_mask, sigma, update, weight = _loss_inputs()
    reference = diffuser.cal_loss(
        x0=x0, x_input=x_input, x_update=update, sigma=sigma, mask=x_mask, atom_weight=weight
    )
    got = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, weight)
    assert abs(float(reference) - float(got)) < 1e-4 * abs(float(reference))


def test_af3_alignment_weights_equal_the_weighted_kabsch_reference(gs):
    """``align_atom_weight=True`` = the repo's SVD alignment with weights ``mask * w_l`` (AF3 Algorithm 28), same loss otherwise."""
    diffuser, x0, x_input, x_mask, sigma, update, weight = _loss_inputs()
    sch = diffuser.scheduler
    got = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, weight, align_atom_weight=True)

    noisy = x_input / sch.input_scale(sigma)
    x_pred = sch.skip_scale(sigma) * noisy + sch.output_scale(sigma) * update
    mask = x_mask.expand(x_pred.shape[:-1])
    x0_safe, pred_safe = torch.where(mask[..., None], x0, 0.0), torch.where(mask[..., None], x_pred, 0.0)
    aligned = weighted_align(x0_safe, pred_safe, weight=weight.expand(mask.shape) * mask)
    sq = torch.where(mask[..., None], (pred_safe - aligned).pow(2), 0.0)
    w = sch.loss_weight(sigma) * mask[..., None] * weight[..., None]
    reference = ((sq * w).sum((-2, -1)) / (mask.sum(-1).clamp_min(1) * 3)).mean()
    assert abs(float(reference) - float(got)) < 1e-4 * abs(float(reference))


def test_atom_weighted_alignment_minimises_the_weighted_error_and_differs_from_the_mask_only_one(gs):
    diffuser, x0, x_input, x_mask, sigma, update, weight = _loss_inputs()
    ones = torch.ones_like(weight)
    mask_only = _weighted_sq_error(diffuser, x0, x_input, x_mask, sigma, update, weight, ones)
    atom_weighted = _weighted_sq_error(diffuser, x0, x_input, x_mask, sigma, update, weight, weight)
    assert (atom_weighted <= mask_only * (1 + 1e-5)).all()  # Kabsch is optimal for the weights it is given
    assert (atom_weighted < mask_only * 0.9999).any()  # and the two alignments really differ


def test_uniform_atom_weights_give_the_same_loss_with_the_flag_on_or_off(gs):
    diffuser, x0, x_input, x_mask, sigma, update, _ = _loss_inputs()
    ones = torch.ones(1, x0.shape[-2])
    off = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, ones)
    on = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, ones, align_atom_weight=True)
    assert abs(float(off) - float(on)) < 1e-6 * abs(float(off))


def test_atom_weighted_alignment_carries_no_gradient_through_the_alignment(gs):
    diffuser, x0, x_input, x_mask, sigma, update, weight = _loss_inputs()
    update = update.requires_grad_()
    loss = gs.cal_loss_gs(diffuser, x0, x_input, update, sigma, x_mask, weight, align_atom_weight=True)
    (grad,) = torch.autograd.grad(loss, update)
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0


def test_graph_safe_sampling_swaps_and_restores(gs):
    diffuser = _diffuser()
    before = (
        diffuser.random_rotation_and_translation.__func__,
        diffuser.scheduler.sample_noise.__func__,
    )
    with gs.graph_safe_sampling(diffuser):
        assert diffuser.random_rotation_and_translation.__func__ is gs.rot_trans_gs
        assert diffuser.scheduler.sample_noise.__func__ is gs.sample_noise_gs
    assert (
        diffuser.random_rotation_and_translation.__func__,
        diffuser.scheduler.sample_noise.__func__,
    ) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_the_stand_ins_capture_and_draw_fresh_numbers_on_every_replay(gs):
    scheduler = SimpleNamespace(config=EDMScheduler.EDMSchedulerConfig())
    rot_owner = SimpleNamespace(config=SimpleNamespace(translation_noise=1.0))
    x = torch.randn(48, 64, 3, device="cuda")
    y = torch.randn_like(x)
    w = torch.ones(48, 64, device="cuda")

    def work():
        return (
            gs.sample_noise_gs(scheduler, 48),
            gs.rot_trans_gs(rot_owner, x),
            gs.weighted_align_gs(x, y, w),
        )

    work()
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = work()
    graph.replay()
    torch.cuda.synchronize()
    first = [t.clone() for t in out]
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(
        first[0], out[0]
    )  # a new noise level per replay (not frozen into the graph)
    assert not torch.equal(first[1], out[1])  # a new rotation per replay
    assert torch.equal(first[2], out[2])  # the alignment is deterministic
