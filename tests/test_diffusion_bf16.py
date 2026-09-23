"""v2.0.0 bf16 diffusion module: norms stay fp32, everything else bf16, output fp32."""

import torch
from team_gm.modules import DiffusionTransformer

from miniworld.configs import SharedConfig
from miniworld.modules.bias_only_token_dit import BiasOnlyTokenDiT
from miniworld.modules.diffusion_module import DiffusionConditioning, DiffusionModule

NORM_TYPES = (torch.nn.LayerNorm, torch.nn.RMSNorm)


def _small_module(kind):
    shared = SharedConfig(d_single=16, d_single_atom=8, d_single_token=16, d_single_token_input=12,
                          d_pair=8, d_pair_atom=4, d_time=256)
    atom = DiffusionTransformer.Config(d_single=8, d_cond=8, d_pair=4, n_head=2, n_block=1)
    token = DiffusionTransformer.Config(d_single=16, d_cond=16, d_pair=8, n_head=2, n_block=2)
    return DiffusionModule(shared, atom, token, DiffusionConditioning.Config(), token_dit_kind=kind)


def _norm_param_names(module) -> set[str]:
    """Parameters owned by a normalisation layer -- decided by module TYPE, not by name
    (an engine LayerNorm inside an nn.Sequential is called ``...0.weight``)."""
    names = set()
    for mname, m in module.named_modules():
        if isinstance(m, NORM_TYPES):
            for pname, _ in m.named_parameters(recurse=False):
                names.add(f"{mname}.{pname}" if mname else pname)
    return names


def test_bf16_cast_keeps_every_norm_affine_in_fp32():
    dm = _small_module("bias_only").to(torch.bfloat16)
    norm_params = _norm_param_names(dm)
    fp32, bf16, wrong = [], [], []
    for name, prm in dm.named_parameters():
        (fp32 if prm.dtype == torch.float32 else bf16).append(name)
        expected = torch.float32 if name in norm_params else torch.bfloat16
        if prm.dtype != expected:
            wrong.append((name, str(prm.dtype)))
    assert norm_params <= set(fp32), "every norm affine param must be fp32"
    assert bf16, "the DiT must actually be in bf16"
    assert fp32, "norm affine params must remain (they are what stays fp32)"
    assert not wrong, f"dtype policy violated for: {wrong[:8]}"
    # the patched engine AdaLN specifically
    adaln = [n for n in fp32 if n.endswith("ada_ln_in.ln_cond.weight")]
    assert adaln, "AdaLN ln_cond.weight must be present and fp32"


def test_bias_only_token_dit_runs_in_bf16_with_fp32_norms():
    torch.manual_seed(0)
    dit = BiasOnlyTokenDiT(BiasOnlyTokenDiT.Config(d_single=32, d_cond=16, d_pair=8, n_head=4, n_block=2)
                           ).to(torch.bfloat16)
    A, B, L = 2, 1, 10
    single = torch.randn(A, B, L, 32).bfloat16().requires_grad_(True)
    cond = torch.randn(A, B, L, 16).bfloat16()
    pair = torch.randn(B, L, L, 8).bfloat16()
    for blk in dit.blocks:
        assert blk.attention.ln_pair.weight.dtype == torch.float32
        assert blk.attention.ada_ln_in.ln_cond.weight.dtype == torch.float32
        assert blk.attention.to_value.weight.dtype == torch.bfloat16
    out = dit(single, cond, pair)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert torch.isfinite(single.grad).all()
    p = dit.blocks[0].attention.attention_pattern(pair)
    assert p.dtype == torch.float32, "the attention softmax is taken in fp32"


def test_v200_configs_select_bf16_and_bias_only():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld"
    for name in ("phase2a_diffusion_v200", "phase2b_diffusion_v200"):
        with initialize_config_dir(str(root), version_base=None):
            cfg = compose(config_name=name)
        assert cfg.model.diffusion.dtype == "bf16" and cfg.model.diffusion.token_dit_kind == "bias_only"
    from miniworld.models.diffusion.model import DiffusionModel
    assert DiffusionModel.DiffusionConfig.model_fields["dtype"].default == "fp32"


def _mock_harness():
    """Reuse the equivalence test's mock batch / t_emb builders (imported by path)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "eq_test", Path(__file__).resolve().parent / "test_inference_equivalence.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod._make_mock_batch, mod._canonical_t_emb


def _cast_like_the_model(dm, dtype):
    """The exact cast policy DiffusionModel applies for ``diffusion.dtype``."""
    dm = dm.to(dtype)
    dm.diffusion_conditioning.relative_position_embedder.float()
    return dm


def _run_module(dm, *, L_token=4, L_atom=9, A=2):
    make_batch, canonical_t_emb = _mock_harness()
    g = torch.Generator().manual_seed(11)
    batch = make_batch(L_token=L_token, L_atom=L_atom, seed=3)
    d_single = dm.add_single_token_cond[0].normalized_shape[0]
    d_in = dm.diffusion_conditioning.linear_token_single[0].normalized_shape[0] - d_single
    d_pair = dm.diffusion_transformer.config.d_pair
    # the module's compute dtype = any Linear weight (the first parameter of the token DiT is
    # an fp32-pinned norm gamma, which is exactly what must NOT decide the input dtype)
    dt = dm.diffusion_conditioning.linear_token_pair[1].weight.dtype
    with torch.no_grad():
        return dm(
            reference=batch.reference, scheme=batch.scheme, structure=batch.structure,
            x_t=torch.randn(A, 1, L_atom, 3, generator=g),                # fp32, as the diffuser gives it
            x_mask=torch.ones(A, 1, L_atom, dtype=torch.bool),
            t_emb=canonical_t_emb(4.0),                                    # fp32
            token_single_input=torch.randn(1, L_token, d_in, generator=g).to(dt),
            token_single_trunk=torch.randn(1, L_token, d_single, generator=g).to(dt),
            token_pair_trunk=torch.randn(1, L_token, L_token, d_pair, generator=g).to(dt),
        )


def test_full_module_forward_runs_in_bf16_af3_atom_path():
    dm = _cast_like_the_model(_small_module("bias_only"), torch.bfloat16).eval()
    out = _run_module(dm)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all() and out.shape[-1] == 3


def test_full_module_forward_runs_in_bf16_swa_atom_path():
    from team_gm.modules import SWAAtomTransformer
    # d_single_atom=64 / n_head=2 -> head_dim 32 >= the 16 active 3D-RoPE frequencies
    shared = SharedConfig(d_single=16, d_single_atom=64, d_single_token=16, d_single_token_input=12,
                          d_pair=8, d_pair_atom=4, d_time=256)
    atom = DiffusionTransformer.Config(d_single=64, d_cond=64, d_pair=4, n_head=2, n_block=1)
    token = DiffusionTransformer.Config(d_single=16, d_cond=16, d_pair=8, n_head=2, n_block=2)
    swa = SWAAtomTransformer.Config(d_atom=64, d_cond=64, n_block=1, n_head=2, swa_window_size=4)
    dm = DiffusionModule(shared, atom, token, DiffusionConditioning.Config(),
                         swa_atom_config=swa, token_dit_kind="bias_only")
    dm = _cast_like_the_model(dm, torch.bfloat16).eval()
    out = _run_module(dm)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
