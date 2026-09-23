"""v2.0.0 BiasOnlyTokenDiT: per-block pattern, masking, gradients, checkpoint parity, wiring."""

import torch
from team_gm.modules.exceptions import ImplementationType

from miniworld.modules.bias_only_token_dit import BiasOnlyTokenDiT

A, B, L, D, C, P, H = 2, 1, 12, 32, 16, 8, 4


def _dit(n_block=3, ckpt=None):
    torch.manual_seed(0)
    return BiasOnlyTokenDiT(BiasOnlyTokenDiT.Config(
        d_single=D, d_cond=C, d_pair=P, n_head=H, n_block=n_block,
        n_checkpoint_segments=ckpt, implementation=ImplementationType.PYTORCH))


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(A, B, L, D, generator=g), torch.randn(A, B, L, C, generator=g),
            torch.randn(B, L, L, P, generator=g))


def test_output_shape_and_pattern_is_a_masked_softmax():
    dit = _dit(); single, cond, pair = _inputs()
    mask = torch.ones(B, L, dtype=torch.bool); mask[:, -3:] = False
    out = dit(single, cond, pair, mask=mask)
    assert out.shape == (A, B, L, D) and torch.isfinite(out).all()
    for blk in dit.blocks:
        p = blk.attention.attention_pattern(pair, mask)
        assert p.shape == (B, H, L, L) and p.dtype == torch.float32
        assert torch.allclose(p.sum(-1), torch.ones(B, H, L))
        assert torch.all(p[..., -3:] == 0), "masked keys must receive zero probability"


def test_each_block_owns_its_pattern_and_has_no_qk(monkeypatch):
    dit = _dit(n_block=3); single, cond, pair = _inputs()
    calls = []
    for i, blk in enumerate(dit.blocks):
        orig = blk.attention.to_bias.forward
        monkeypatch.setattr(blk.attention.to_bias, "forward",
                            lambda x, i=i, orig=orig: calls.append(i) or orig(x))
    dit(single, cond, pair)
    assert calls == [0, 1, 2], "every block projects the pair bias exactly once per forward"
    # the pattern depends on pair only: same pair -> same pattern regardless of single
    blk0 = dit.blocks[0].attention
    assert torch.equal(blk0.attention_pattern(pair), blk0.attention_pattern(pair))
    # blocks can differ from one another once their to_bias weights differ
    with torch.no_grad():
        dit.blocks[1].attention.to_bias.weight.normal_()
    assert not torch.allclose(blk0.attention_pattern(pair), dit.blocks[1].attention.attention_pattern(pair))
    for blk in dit.blocks:
        names = set(dict(blk.attention.named_parameters()))
        assert not any(k.startswith(("to_query", "to_key", "norm_query", "norm_key")) for k in names), names
        assert {"ln_pair.weight", "to_bias.weight", "to_value.weight", "to_gate.weight",
                "to_out.weight", "to_scale.weight", "to_scale.bias"} <= names, names
        assert "ln_pair.bias" not in names


def test_all_parameters_receive_finite_gradients():
    dit = _dit(); single, cond, pair = _inputs()
    # to_out is zero-initialised (as in v1), so at init every attention delta is 0 and the
    # shared bias sees no gradient; perturb to_out to test the path, not the init.
    with torch.no_grad():
        for blk in dit.blocks:
            blk.attention.to_out.weight.normal_(std=0.02)
    single.requires_grad_(True)
    dit(single, cond, pair).pow(2).mean().backward()
    assert torch.isfinite(single.grad).all()
    for n, prm in dit.named_parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all(), n
    for blk in dit.blocks:
        g = blk.attention.to_bias.weight.grad
        assert g is not None and g.abs().sum() > 0, "each block's pair projection must learn"


def test_checkpointed_stack_matches_plain_stack():
    single, cond, pair = _inputs()
    plain, ck = _dit(n_block=4), _dit(n_block=4, ckpt=4)
    ck.load_state_dict(plain.state_dict())
    torch.testing.assert_close(plain(single, cond, pair), ck(single, cond, pair), rtol=1e-5, atol=1e-5)


def test_augment_axis_shares_the_pattern():
    dit = _dit(); single, cond, pair = _inputs()
    # feeding augment 0 alone must equal augment 0 of the batched call
    full = dit(single, cond, pair)
    solo = dit(single[:1], cond[:1], pair)
    torch.testing.assert_close(full[:1], solo, rtol=1e-5, atol=1e-5)


def test_diffusion_config_switch_defaults_to_v1():
    from miniworld.models.diffusion.model import DiffusionModel
    fields = DiffusionModel.DiffusionConfig.model_fields
    assert fields["token_dit_kind"].default == "augmented"


def test_v200_configs_compose_and_select_bias_only():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld"
    for name in ("phase2a_diffusion_v200", "phase2b_diffusion_v200"):
        with initialize_config_dir(str(root), version_base=None):
            cfg = compose(config_name=name)
        assert cfg.model.diffusion.token_dit_kind == "bias_only"
        assert cfg.model.diffusion.token_dit.n_block == 24
        assert "v2.0.0" in cfg.train.run_dir
