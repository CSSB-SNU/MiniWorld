"""The capture-safe stand-ins of ``scripts/phase2_speed/graph_safe.py`` against the repo's own host-synchronising versions."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from team_gm.diffusion import EDMScheduler, EuclideanDiffuser
from team_gm.utils.align import weighted_align

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gs():
    path = str(ROOT / "scripts" / "phase2_speed")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("graph_safe")
    finally:
        sys.path.remove(path)


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
