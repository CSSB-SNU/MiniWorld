"""Static caches built once per ``prepare`` / once per ``sample``.

Two dataclasses:

- :class:`InferenceCache` holds everything the diffusion module reuses
  across **every** step and **every** sample of one ``prepare(batch)``
  call: trunk outputs, the t-independent half of
  ``DiffusionConditioning``, the fully baked ``atom_pair`` tensor (the
  costliest hoist — see ``build_inference_cache`` for the audit), the
  atom-to-token scatter scaffolding, etc.

- :class:`StepSchedule` holds everything that depends only on the
  diffusion timestep schedule (i.e. on ``num_steps`` and optionally
  ``start_sigma_y``). Once built, lookups inside the per-step kernel are
  pure indexing.

Both are deliberately ``frozen=True`` to make accidental in-place
mutation a loud bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from miniworld.data.features.batch import Batch
    from miniworld.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler
    from miniworld.diffusion.decoupled_xpred.solver import XPredDecoupledSolver
    from miniworld.models.miniworld.model import Model


@dataclass(frozen=True)
class InferenceCache:
    """Per-batch static cache.

    Built once by :meth:`Predictor.prepare`. Independent of:
      - the diffusion timestep ``t``
      - the diffusion sample index along the augmentation axis ``A``
      - the noisy coordinates ``x_t``

    Reused by every solver step and every sample chunk for the same batch.
    """

    # --- Original batch reference (held so the solver can read scheme/structure
    # for R/T sampling and ODE bookkeeping without re-threading every field).
    batch: "Batch"

    # --- Trunk outputs (B, ...)
    token_single_input: torch.Tensor          # (B, L_token, d_single_token_input)
    token_single_trunk: torch.Tensor          # (B, L_token, d_single)
    token_pair_trunk: torch.Tensor            # (B, L_token, L_token, d_pair)
    distogram_logit: torch.Tensor             # (B, L, L, n_distogram_bins)

    # --- DiffusionConditioning, t-independent half
    token_pair_cond: torch.Tensor             # (B, L_token, L_token, d_pair)
    token_single_pre_time: torch.Tensor       # (B, L_token, d_single)

    # --- AtomAttentionEncoder, t/x_t-independent half
    # ``atom_single_cond_base`` already folds in the token->atom gather of
    # ``token_single_to_atom_single_cond(token_single_trunk)``; per step we
    # only need to add ``noisy_to_atom_single_rep(x_t) * x_mask`` on top.
    atom_single_cond_base: torch.Tensor       # (B, L_atom, d_single_atom)
    # ``atom_pair`` is the costly hoist: post ``mlp_atom_pair`` so each step
    # skips an [B, L_atom, L_atom, d_pair_atom] MLP pass.
    atom_pair: torch.Tensor                   # (B, L_atom, L_atom, d_pair_atom)

    # --- Scatter scaffolding for ``_scatter_atom_to_token``
    scatter_mapping: torch.Tensor             # (B, L_atom, L_token) one-hot fp
    scatter_count_inv: torch.Tensor           # (B, L_token, 1) 1.0/clamp(count, 1)

    # --- Masks the per-step kernel needs in pre-expanded form
    atom_mask: torch.Tensor                   # (B, L_atom) bool
    token_mask: torch.Tensor                  # (B, L_token) bool


@dataclass(frozen=True)
class StepSchedule:
    """Per-timestep precomputed values, sized along the leading T axis.

    ``T = num_steps``. The last step is the one that lands on
    ``time_steps[-1] == 0``.
    """

    # Solver-side scalars (all shape (T,))
    sigma_i: torch.Tensor                     # noise level at start of step (pre-injection)
    sigma_hat: torch.Tensor                   # sigma_i * (1 + gamma) — post-injection
    sigma_next: torch.Tensor                  # sigma at end of step
    sigma_t_hat: torch.Tensor                 # translation-noise scale at sigma_hat
    c_in: torch.Tensor                        # input scaling 1/sqrt(sig^2 + sig_t^2 + sig_data^2)
    gamma: torch.Tensor                       # per-step stochastic-injection scale
    noise_scale: torch.Tensor                 # x0_centered update rule's residual noise scale

    # Precomputed conditioning per step
    # ``token_single_cond[t]`` is what ``DiffusionConditioning(... t_emb=t_emb_t ...)``
    # would have returned for the single branch; we run it once for all T.
    token_single_cond: torch.Tensor           # (T, B, L_token, d_single)
    # ``added_token_cond[t]`` is ``add_single_token_cond(token_single_cond[t])``;
    # cached so the per-step kernel just adds a constant.
    added_token_cond: torch.Tensor            # (T, B, L_token, d_single_token)

    # Original schedule, for downstream tooling (distogram plots etc.)
    time_steps: torch.Tensor                  # (T+1,)


def build_inference_cache(
    model: "Model",
    batch: "Batch",
    *,
    token_single_input: torch.Tensor,
    token_single_trunk: torch.Tensor,
    token_pair_trunk: torch.Tensor,
    distogram_logit: torch.Tensor,
) -> InferenceCache:
    """Build the per-batch static cache from the trunk outputs.

    Caller (Predictor) is responsible for running the trunk and passing
    its outputs in. We do not run the trunk here so this function stays
    pure / testable.
    """
    dm = model.diffusion_module
    cond = dm.diffusion_conditioning
    enc = dm.atom_attention_encoder

    scheme = batch.scheme
    reference = batch.reference
    structure = batch.structure

    # --- DiffusionConditioning pair branch (t-independent) ---
    rel_emb = cond.relative_position_embedder(
        asym_id=scheme.token_asym_id,
        token_residue_idx=scheme.token_residue_idx,
        token_idx=scheme.token_idx,
        entity_id=scheme.token_entity_id,
        sym_id=scheme.token_sym_id,
    )
    token_pair_cond = torch.cat([token_pair_trunk, rel_emb], dim=-1)
    token_pair_cond = cond.linear_token_pair(token_pair_cond)
    for transition in cond.pair_transitions:
        token_pair_cond = token_pair_cond + transition(token_pair_cond)

    # --- DiffusionConditioning single branch, pre-time (t-independent) ---
    token_single_pre_time = cond.linear_token_single(
        torch.cat([token_single_input, token_single_trunk], dim=-1),
    )

    # --- AtomAttentionEncoder, t/x_t-independent half ---
    # We mirror ``AtomAttentionEncoder._before_atom_transformer`` but stop
    # before the ``atom_single_rep = atom_single_cond + noisy(x_t)`` step
    # and the ``atom_transformer`` call. We also stop after the final
    # ``mlp_atom_pair`` so ``atom_pair`` is fully baked.

    # ``init_atom_features`` is pure-data and has its own no_grad guard.
    from miniworld.modules.diffusion_module import init_atom_features
    atom_single_init, atom_pair_init = init_atom_features(reference)

    atom_single_cond = enc.to_atom_single_cond(atom_single_init)
    atom_pair = enc.to_atom_pair(atom_pair_init)

    device = atom_single_cond.device
    batch_size, atom_length = atom_single_cond.shape[:2]

    _to_add_single = enc.token_single_to_atom_single_cond(token_single_trunk)
    _to_add_pair = enc.token_pair_to_atom_pair(token_pair_cond)

    atom_to_token_idx_map = scheme.atom_to_token_idx_map
    batch_1d_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(
        -1, atom_length,
    )
    atom_single_cond = (
        atom_single_cond + _to_add_single[batch_1d_idx, atom_to_token_idx_map]
    )

    # Canonical pair gather collapses to the per-(b, j) column-atom token id
    # (see the long comment in diffusion_module._before_atom_transformer_chunked
    # for why) — gather as [B, L_atom, d_pair_atom] and broadcast over rows.
    b_arange = torch.arange(batch_size, device=device)
    diag_gather = _to_add_pair[
        b_arange.unsqueeze(1),
        atom_to_token_idx_map,
        atom_to_token_idx_map,
    ]
    _left = enc.atom_single_to_pair_left(atom_single_cond)
    _right = enc.atom_single_to_pair_right(atom_single_cond)
    atom_pair = atom_pair + diag_gather.unsqueeze(1) + _left.unsqueeze(2) + _right.unsqueeze(1)
    atom_pair = atom_pair + enc.mlp_atom_pair(atom_pair)

    # --- Scatter scaffolding ---
    token_length = int(scheme.token_idx.shape[1])
    atom_mask = structure.atom_mask
    mapping = torch.nn.functional.one_hot(
        atom_to_token_idx_map,
        num_classes=token_length,
    ).to(atom_single_cond.dtype)              # (B, L_atom, L_token)
    mask_f = atom_mask.to(atom_single_cond.dtype)
    count = torch.einsum("bal,ba->bl", mapping, mask_f)
    scatter_count_inv = 1.0 / count.unsqueeze(-1).clamp(min=1.0)

    return InferenceCache(
        batch=batch,
        token_single_input=token_single_input,
        token_single_trunk=token_single_trunk,
        token_pair_trunk=token_pair_trunk,
        distogram_logit=distogram_logit,
        token_pair_cond=token_pair_cond,
        token_single_pre_time=token_single_pre_time,
        atom_single_cond_base=atom_single_cond,
        atom_pair=atom_pair,
        scatter_mapping=mapping,
        scatter_count_inv=scatter_count_inv,
        atom_mask=atom_mask.bool(),
        token_mask=structure.token_mask.bool(),
    )


def build_step_schedule(
    model: "Model",
    cache: InferenceCache,
    scheduler: "DecoupledXPredScheduler",
    solver: "XPredDecoupledSolver",
    *,
    num_steps: int,
    start_sigma_y: float | None = None,
) -> StepSchedule:
    """Build the per-step schedule (sigmas + precomputed token cond).

    ``solver`` is taken purely to read its ``gamma_0`` / ``gamma_min`` /
    ``step_scale`` config — we do not actually call its ``step`` here.
    """
    if start_sigma_y is not None:
        time_steps = scheduler.sampling_time_steps(num_steps, start_sigma_y=start_sigma_y)
    else:
        time_steps = scheduler.sampling_time_steps(num_steps)
    device = cache.token_single_pre_time.device
    time_steps = time_steps.to(device)

    # Per-step sigma scalars
    sigma_i_list = [scheduler.sampling_schedule(time_steps[t]) for t in range(num_steps)]
    sigma_next_list = [
        scheduler.sampling_schedule(time_steps[t + 1]) for t in range(num_steps)
    ]
    sigma_i = torch.stack(sigma_i_list, dim=0).reshape(num_steps)
    sigma_next = torch.stack(sigma_next_list, dim=0).reshape(num_steps)

    gamma_0 = solver.config.gamma_0
    gamma_min = solver.config.gamma_min
    gamma = torch.where(
        sigma_next > gamma_min,
        torch.full_like(sigma_i, float(gamma_0)),
        torch.zeros_like(sigma_i),
    )
    sigma_hat = sigma_i * (1.0 + gamma)
    _, sigma_t_hat = scheduler.convert_to_sigma_rt(sigma_hat)
    c_in = scheduler.input_scale(sigma_hat, sigma_t_hat)

    step_scale = solver.config.step_scale
    noise_scale = (1.0 - step_scale) * sigma_hat + step_scale * sigma_next

    # --- Precompute token_single_cond[t] and added_token_cond[t] ---
    dm = model.diffusion_module
    cond = dm.diffusion_conditioning
    base = cache.token_single_pre_time            # (B, L_token, d_single)

    token_single_cond_list = []
    added_token_cond_list = []
    for t in range(num_steps):
        sigma_t = sigma_hat[t]
        # Existing trunk path uses t_emb shape (1,1,1,1) so the final
        # broadcast adds a leading singleton axis. We replicate that by
        # passing the scalar through noise_condition then fourier_embedding
        # with a single trailing singleton — yields (1, d_time) which feeds
        # the LayerNorm+Linear in add_time_embedding.
        t_emb_t = scheduler.noise_condition(sigma_t).reshape(1)
        from miniworld.modules.embeddings import fourier_embedding
        time_embedding = fourier_embedding(t_emb_t)     # (1, d_time)
        single = base + cond.add_time_embedding(time_embedding)   # (B, L_token, d_single)
        for trans in cond.single_transitions:
            single = single + trans(single)
        single = cond.final_layernorm_token_single(single)
        token_single_cond_list.append(single)
        added_token_cond_list.append(dm.add_single_token_cond(single))

    token_single_cond = torch.stack(token_single_cond_list, dim=0)    # (T, B, L, d_single)
    added_token_cond = torch.stack(added_token_cond_list, dim=0)      # (T, B, L, d_single_token)

    return StepSchedule(
        sigma_i=sigma_i,
        sigma_hat=sigma_hat,
        sigma_next=sigma_next,
        sigma_t_hat=sigma_t_hat,
        c_in=c_in,
        gamma=gamma,
        noise_scale=noise_scale,
        token_single_cond=token_single_cond,
        added_token_cond=added_token_cond,
        time_steps=time_steps,
    )
