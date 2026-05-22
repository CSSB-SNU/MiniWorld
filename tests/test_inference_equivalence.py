"""Equivalence test for ``miniworld.inference`` vs the canonical diffusion path.

The inference path hoists most of ``DiffusionModule.forward`` into a
:class:`~miniworld.inference.cache.InferenceCache` built once per batch;
this test pins down that the cached path produces numerically the same
``atom_pos_update`` for the same ``(reference, scheme, structure,
token_single_input, token_single_trunk, token_pair_trunk, x_t, t_emb)``.

We build a small ``DiffusionModule`` with random weights and feed it a
mock batch — the canonical ``DiffusionModule.forward`` only touches a
narrow slice of ``Batch`` (scheme/reference/structure), so a
``SimpleNamespace`` mock is enough to exercise the equivalence without
spinning up the full ``Model`` / ``Batch`` machinery.

Run:
    pytest libs/MiniWorld/tests/test_inference_equivalence.py -v
or:
    python libs/MiniWorld/tests/test_inference_equivalence.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from team_gm.modules import DiffusionTransformer

from miniworld.configs import SharedConfig
from miniworld.inference.cache import StepSchedule, build_inference_cache
from miniworld.inference.diffusion import diffusion_step
from miniworld.modules.diffusion_module import (
    DiffusionConditioning,
    DiffusionModule,
)
from miniworld.modules.embeddings import fourier_embedding


def _build_diffusion_module(
    d_single: int = 16,
    d_single_atom: int = 8,
    d_single_token: int = 16,
    d_single_token_input: int = 12,
    d_pair: int = 8,
    d_pair_atom: int = 4,
    # ``fourier_embedding`` uses a hardcoded 256-element weight/bias table
    # (see embeddings.py), so d_time is effectively pinned to 256 — the
    # module's add_time_embedding LayerNorm normalizes that dim.
    d_time: int = 256,
) -> DiffusionModule:
    """Construct a small DiffusionModule with non-zero random weights."""
    shared = SharedConfig(
        d_single=d_single,
        d_single_atom=d_single_atom,
        d_single_token=d_single_token,
        d_single_token_input=d_single_token_input,
        d_pair=d_pair,
        d_pair_atom=d_pair_atom,
        d_time=d_time,
    )
    atom_dit = DiffusionTransformer.Config(
        d_single=d_single_atom,
        d_cond=d_single_atom,
        d_pair=d_pair_atom,
        n_head=2,
        n_block=1,
    )
    token_dit = DiffusionTransformer.Config(
        d_single=d_single_token,
        d_cond=d_single,
        d_pair=d_pair,
        n_head=2,
        n_block=1,
    )
    dit_cond = DiffusionConditioning.Config()

    dm = DiffusionModule(shared, atom_dit, token_dit, dit_cond)
    dm.eval()

    # Force non-zero init so the residual / zero-init layers (e.g. the last
    # linear of mlp_atom_pair or add_single_token_cond) actually carry signal.
    # Without this the equivalence holds trivially because both paths produce
    # all-zero intermediates.
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in dm.parameters():
            if p.numel() == 0:
                continue
            p.normal_(mean=0.0, std=0.3, generator=g)
    return dm


def _make_mock_batch(
    *,
    L_token: int,
    L_atom: int,
    seed: int = 0,
) -> SimpleNamespace:
    """A SimpleNamespace mock that satisfies the fields the inference path reads."""
    g = torch.Generator().manual_seed(seed)
    base_map = torch.tensor(
        [i % L_token for i in range(L_atom)], dtype=torch.long,
    ).unsqueeze(0)
    scheme = SimpleNamespace(
        token_asym_id=torch.zeros(1, L_token, dtype=torch.long),
        token_residue_idx=torch.arange(L_token, dtype=torch.long).unsqueeze(0),
        token_idx=torch.arange(L_token, dtype=torch.long).unsqueeze(0),
        token_entity_id=torch.zeros(1, L_token, dtype=torch.long),
        token_sym_id=torch.zeros(1, L_token, dtype=torch.long),
        atom_to_token_idx_map=base_map,
        atom_to_chain_id=torch.zeros(1, L_atom, dtype=torch.long),
    )
    reference = SimpleNamespace(
        pos=torch.randn(1, L_atom, 3, generator=g),
        mask=torch.ones(1, L_atom),
        element=torch.randint(0, 10, (1, L_atom), generator=g).float(),
        charge=torch.randn(1, L_atom, generator=g),
        space_uid=torch.zeros(1, L_atom, dtype=torch.long),
    )
    structure = SimpleNamespace(
        atom_mask=torch.ones(1, L_atom, dtype=torch.bool),
        atom_pos=reference.pos.clone(),
        atom_pos_mask=torch.ones(1, L_atom, dtype=torch.bool),
        token_mask=torch.ones(1, L_token, dtype=torch.bool),
    )
    return SimpleNamespace(scheme=scheme, reference=reference, structure=structure)


def _build_step_schedule_for_single_t(
    dm: DiffusionModule,
    *,
    token_single_input: torch.Tensor,
    token_single_trunk: torch.Tensor,
    token_pair_trunk: torch.Tensor,
    rel_emb_token_pair_cond: torch.Tensor,  # the cached token_pair_cond
    scheme,
    sigma_hat: float,
) -> StepSchedule:
    """Single-timestep StepSchedule whose token_single_cond matches what
    ``DiffusionConditioning.forward`` produces for the given ``sigma_hat``.

    We bypass the real scheduler (which lives in the matplotlib-pulling
    miniworld.diffusion package — see test_inference_equivalence README)
    and instead run the cond's single branch by hand for one timestep.
    """
    cond = dm.diffusion_conditioning

    sigma_t_tensor = torch.tensor([sigma_hat], dtype=torch.float32)
    # Match the noise_condition formula used by DecoupledXPredScheduler.
    t_emb_scalar = sigma_t_tensor.log() / 4.0
    # Build the (B=1, L_token, d_single) single-cond tensor — same ops as in
    # build_step_schedule but for a single t.
    time_embedding = fourier_embedding(t_emb_scalar)            # (1, d_time)
    pre_time = cond.linear_token_single(
        torch.cat([token_single_input, token_single_trunk], dim=-1),
    )
    single = pre_time + cond.add_time_embedding(time_embedding)
    for trans in cond.single_transitions:
        single = single + trans(single)
    single = cond.final_layernorm_token_single(single)

    # (T=1, B, L_token, d_single) and (T=1, B, L_token, d_single_token)
    token_single_cond_stack = single.unsqueeze(0)
    added_token_cond_stack = dm.add_single_token_cond(single).unsqueeze(0)

    zeros = torch.zeros(1)
    return StepSchedule(
        sigma_i=sigma_t_tensor,
        sigma_hat=sigma_t_tensor,
        sigma_next=sigma_t_tensor,
        sigma_t_hat=zeros,
        c_in=zeros,
        gamma=zeros,
        noise_scale=zeros,
        token_single_cond=token_single_cond_stack,
        added_token_cond=added_token_cond_stack,
        time_steps=torch.tensor([float(sigma_hat), 0.0]),
    )


def _canonical_t_emb(sigma_hat: float) -> torch.Tensor:
    """Build the (A=1, B=1, 1, 1)-shape t_emb that ModelWrapper feeds into DiffusionModule.

    ModelWrapper.forward does ``t_emb[None, None, None, None]`` on a scalar
    noise level, so the trunk path actually sees a 4-D tensor whose values
    broadcast over (A, B, L_token, d_time). The noise level itself goes
    through scheduler.noise_condition = log/4 first.
    """
    sigma = torch.tensor(float(sigma_hat), dtype=torch.float32)
    t = sigma.log() / 4.0
    return t[None, None, None, None]


def _assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    name: str,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        msg = (
            f"{name}: inference vs canonical mismatch — "
            f"max_abs_diff={diff.max().item():.3e} "
            f"mean_abs_diff={diff.mean().item():.3e} "
            f"shape={tuple(a.shape)}"
        )
        raise AssertionError(msg)


def test_diffusion_step_matches_canonical() -> None:
    """Single solver step: canonical DiffusionModule.forward vs cache + diffusion_step."""
    dm = _build_diffusion_module()
    shared = dm.diffusion_conditioning.relative_position_embedder
    # Read effective dims off the constructed module rather than reach into config.
    d_single = dm.add_single_token_cond[0].normalized_shape[0]
    d_single_token_input = (
        dm.diffusion_conditioning.linear_token_single[0].normalized_shape[0] - d_single
    )
    d_pair = dm.diffusion_transformer.config.d_pair

    L_token, L_atom = 4, 9
    g = torch.Generator().manual_seed(11)
    batch = _make_mock_batch(L_token=L_token, L_atom=L_atom, seed=3)
    token_single_input = torch.randn(1, L_token, d_single_token_input, generator=g)
    token_single_trunk = torch.randn(1, L_token, d_single, generator=g)
    token_pair_trunk = torch.randn(1, L_token, L_token, d_pair, generator=g)

    # An arbitrary noise level. sigma>1 so log/4 is positive.
    sigma_hat = 4.0
    t_emb_canonical = _canonical_t_emb(sigma_hat)

    A = 2  # exercise the augmentation axis
    x_t_AB = torch.randn(A, 1, L_atom, 3, generator=g)  # (A, B=1, L_atom, 3)
    x_mask_AB = torch.ones(A, 1, L_atom, dtype=torch.bool)

    # --- Canonical path ---
    with torch.inference_mode():
        out_canonical = dm.forward(
            reference=batch.reference,
            scheme=batch.scheme,
            structure=batch.structure,
            x_t=x_t_AB,
            x_mask=x_mask_AB,
            t_emb=t_emb_canonical,
            token_single_input=token_single_input,
            token_single_trunk=token_single_trunk,
            token_pair_trunk=token_pair_trunk,
        )

    # --- Inference path ---
    fake_model = SimpleNamespace(diffusion_module=dm)
    with torch.inference_mode():
        cache = build_inference_cache(
            fake_model,
            batch,
            token_single_input=token_single_input,
            token_single_trunk=token_single_trunk,
            token_pair_trunk=token_pair_trunk,
            distogram_logit=torch.zeros(1, L_token, L_token, 1),
        )
        schedule = _build_step_schedule_for_single_t(
            dm,
            token_single_input=token_single_input,
            token_single_trunk=token_single_trunk,
            token_pair_trunk=token_pair_trunk,
            rel_emb_token_pair_cond=cache.token_pair_cond,
            scheme=batch.scheme,
            sigma_hat=sigma_hat,
        )

        x_t_A = x_t_AB.squeeze(1)              # (A, L_atom, 3)
        out_inference_A = diffusion_step(
            fake_model, cache, schedule, x_t_A, t_index=0,
        )
        # Reinject the B=1 axis dropped by diffusion_step so shapes line up
        # with the canonical (A, B, L_atom, 3) return.
        out_inference = out_inference_A.unsqueeze(1)

    # Atom-attention is masked: with all-ones mask the canonical path still
    # has a tiny bit of float32 reordering vs the cached path because the
    # encoder's atom_transformer runs on slightly different intermediates
    # (cache pre-materialises atom_pair). atol=1e-5 has held in spot checks.
    _assert_close(out_inference, out_canonical, "atom_pos_update", atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    for fn in (test_diffusion_step_matches_canonical,):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            raise
