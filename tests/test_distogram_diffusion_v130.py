"""v1.3 EDM target, weighting, sampling, wrapper wiring and graph RNG regressions."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from team_gm.modules.layers.embeddings import fourier_embedding

from miniworld.loss.auxiliary import atom_distogram_target, cal_atom_distogram_loss
from miniworld.models.distogram_only.client import Client
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.modules.distogram_diffusion import (
    DistogramDiffusion, DistogramDiffusionConfig, _symmetric_noise,
)


def head(device="cpu"):
    return DistogramDiffusion(8, 96, DistogramDiffusionConfig(
        bin_center=85.7, scheduler={"sigma_data": 18.9},
    )).float().to(device)


def test_weight_doubles_only_interchain_gradients_and_uses_unweighted_count():
    h = head()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).requires_grad_()
    target = torch.zeros_like(x)
    mask = torch.ones_like(x, dtype=torch.bool)
    sigma = torch.tensor([2.])
    chains = torch.tensor([[0, 0, 1, 1]])
    plain, _ = h.loss(x, target, mask, sigma, chains, 1.)
    weighted, stats = h.loss(x, target, mask, sigma, chains, 2.)
    g1, = torch.autograd.grad(plain, x, retain_graph=True)
    g2, = torch.autograd.grad(weighted, x)
    cross = chains[:, :, None] != chains[:, None, :]
    torch.testing.assert_close(g2, g1 * torch.where(cross, 2., 1.), rtol=0, atol=0)
    expected = (x.square() * torch.where(cross, 2., 1.)).triu(1).sum() / 6
    torch.testing.assert_close(weighted, expected * h.scheduler.loss_weight(sigma)[0])
    assert all(isinstance(v, torch.Tensor) and not v.requires_grad for v in stats.values())


def test_empty_mask_and_unobserved_nan_do_not_pollute_gradients():
    h = head()
    x = torch.full((2, 3, 3), float("nan"), requires_grad=True)
    loss, stats = h.loss(x, x.detach(), torch.zeros_like(x, dtype=torch.bool), torch.ones(2))
    loss.backward()
    assert loss.item() == 0 and stats["bin_rmse"].item() == 0
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_symmetric_noise_retains_unit_variance():
    torch.manual_seed(10)
    n = _symmetric_noise(256, 24, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(n, n.transpose(-1, -2), rtol=0, atol=0)
    assert n.diagonal(dim1=-2, dim2=-1).count_nonzero() == 0
    values = n[:, torch.ones(24, 24, dtype=torch.bool).triu(1)]
    assert abs(values.var().item() - 1.) < .025


def test_factored_encoder_matches_original_concat_forward_and_gradients():
    torch.manual_seed(2)
    h = head()
    x = torch.randn(2, 5, 5, requires_grad=True)
    sigma = torch.tensor([.5, 12.])
    actual = h.encode_input(x, sigma, torch.float32)
    scaled = (x * h.scheduler.input_scale(sigma)[:, None, None])[..., None]
    time = fourier_embedding(h.scheduler.noise_condition(sigma))[:, None, None].expand(2, 5, 5, -1)
    expected = h.encoder(torch.cat((scaled, time), dim=-1))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, h.encoder.weight), grad, retain_graph=True)
    ge = torch.autograd.grad(expected, (x, h.encoder.weight), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-6, rtol=2e-5)


def test_edm_preconditioning_formula():
    h = head()
    x = torch.randn(2, 3, 3)
    sigma = torch.tensor([.1, 100.])
    pair = torch.randn(2, 3, 3, 8)
    with torch.no_grad(): h.decoder.weight.fill_(.125)
    f = h.decoder(pair)[..., 0]
    f = .5 * (f + f.transpose(-1, -2))
    expected = x * (18.9**2 / (sigma**2 + 18.9**2))[:, None, None]
    expected += f * (sigma * 18.9 / (sigma**2 + 18.9**2).sqrt())[:, None, None]
    torch.testing.assert_close(h.decode_output(pair, x, sigma), expected)


def test_heun_terminal_zero_and_mask_with_oracle_denoiser():
    h = head()
    bins = torch.tensor([[[0, 12, 50], [12, 0, 95], [50, 95, 0]]])
    calls = []
    def oracle(x, sigma):
        calls.append(sigma.clone())
        return h.encode(bins)
    result = h.sample(oracle, 1, 3, torch.device("cpu"), num_steps=4)
    torch.testing.assert_close(result, bins)
    assert len(calls) == 7 and all((s > 0).all() for s in calls)
    result = h.sample(oracle, 1, 3, torch.device("cpu"), num_steps=4,
                      token_mask=torch.tensor([[True, True, False]]))
    assert result[:, 2].count_nonzero() == 0 and result[:, :, 2].count_nonzero() == 0
    with pytest.raises(ValueError): h.sample(oracle, 1, 3, torch.device("cpu"), num_steps=0)


def test_shared_targets_match_ce_including_missing_and_overflow():
    pos = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [100., 0., 0.], [float("nan"), 0., 0.]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    mapping = torch.arange(4)[None]
    bins, valid = atom_distogram_target(pos, mask, mapping, 96, 5, rep_atom_mask=mask)
    logits = torch.randn(1, 5, 5, 96)
    actual = cal_atom_distogram_loss(logits, pos, mask, mapping, rep_atom_mask=mask)
    ce = F.cross_entropy(logits.permute(0, 3, 1, 2), bins, reduction="none")
    valid = valid.triu(1)
    torch.testing.assert_close(actual, (ce * valid).sum((-1, -2)) / valid.sum((-1, -2)))
    assert bins[0, 0, 1] == 0 and bins[0, 0, 2] == 95
    assert not valid[0, 3:].any()


def test_client_calls_wrapped_forward_with_interface_weight():
    seen = {}
    class Wrapped(torch.nn.Module):
        def forward(self, **kwargs):
            seen.update(kwargs)
            value = torch.tensor(3., requires_grad=True)
            return value, {"diffusion_loss": value.detach(), "bin_rmse": torch.tensor(2.)}
    model = Wrapped()
    model.register_forward_hook(lambda *args: seen.update(wrapper_called=True))
    client = NS(model=model, config=NS(
        model=NS(trunk=NS(diffusion=object())),
        loss=NS(distogram_cb_target=True, distogram_loss=.5, distogram_interchain_weight=2.),
    ))
    batch = NS(**dict.fromkeys(("msa", "reference", "scheme", "sequence", "structure", "template")))
    loss, stats = Client.loss_fn(client, batch)
    assert loss.item() == 1.5 and seen["interchain_weight"] == 2. and seen["wrapper_called"]
    assert stats["diffusion_loss"] == 3.


@pytest.mark.parametrize("name,length,atoms", [
    ("distogram_diffusion_medium_v130_bioai", 384, 4096),
    ("distogram_diffusion_medium_v130_bioai_L768", 768, 8192),
])
def test_v130_config_preserves_v120_policies(name, length, atoms):
    root = Path(__file__).resolve().parents[1] / "configs/miniworld"
    with initialize_config_dir(str(root), version_base=None): cfg = compose(config_name=name)
    model = MiniSWAModel.Config.model_validate(cfg.model)
    assert model.trunk.pairformer.n_block == 16 and model.trunk.msa_module.n_block == 4
    assert model.trunk.use_template and model.trunk.n_recycle_max == 1
    assert model.trunk.diffusion is not None and model.shared.n_distogram_bins == 96
    assert model.trunk.msa_subsample_per_recycle == 1024
    assert cfg.data.msa.max_msa_depth_by_source.pdb == 8192
    assert cfg.data.msa.sample_depth_by_source.pdb == "af3"
    assert cfg.data.crop.ab_ag_interface_only and cfg.data.crop.prefer_nonprotein_focus
    assert cfg.loss.distogram_interchain_weight == 2.
    assert cfg.data.crop.max_tokens == cfg.train.bucket_token_multiple == length
    assert cfg.data.crop.max_atoms == cfg.train.bucket_atom_multiple == atoms
    assert cfg.train.force_trainer == "cudagraph" and cfg.train.engine_backend == "auto"
    assert str(cfg.data.train_db.pdb.cif_db_path).startswith("/data/")


@pytest.mark.parametrize("scheduler", [{"sigma_data": 0}, {"rho": -1}, {"P_std": -2}, {"sigma_max": float("nan")}])
def test_invalid_schedule_rejected(scheduler):
    with pytest.raises(ValueError): DistogramDiffusionConfig(scheduler=scheduler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_sigma_and_noise_change_without_host_sampling():
    h = head("cuda")
    def draw():
        sigma = h.sample_sigma(2, torch.device("cuda"))
        return sigma, h.add_noise(torch.zeros(2, 16, 16, device="cuda"), sigma)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): draw()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): sigma, noise = draw()
    g.replay()
    first_sigma, first_noise = sigma.clone(), noise.clone()
    g.replay()
    assert not torch.equal(first_sigma, sigma) and not torch.equal(first_noise, noise)
    torch.testing.assert_close(noise, noise.transpose(-1, -2), rtol=0, atol=0)
