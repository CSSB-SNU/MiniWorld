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
    from team_gm.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler
    from team_gm.diffusion.decoupled_xpred.solver import XPredDecoupledSolver
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
        token_pair_cond = transition(token_pair_cond)

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

    # Pair-cond gather: out[b, i, j] = _to_add_pair[b, A[b, i], A[b, j]].
    # Row/col indices orthogonalized via unsqueeze so they broadcast onto
    # distinct axes. Materialized lazily — the chunked branch below gathers
    # one row-chunk at a time so the temporary is [B, chunk, L_atom, d].
    b_arange = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    col_tokens = atom_to_token_idx_map.unsqueeze(-2)  # [B, 1, L_atom]
    _left = enc.atom_single_to_pair_left(atom_single_cond)
    _right = enc.atom_single_to_pair_right(atom_single_cond)

    # Inference chunking: same env gate as the other _before_atom_transformer
    # paths. Without chunking, the pair gather + broadcast adds + mlp
    # materialise [B, L_atom, L_atom, d] temporaries (~13 GiB at L_atom=14k
    # for H1335), which OOMs cache prep before inference even starts.
    # Chunked atom attention is the default here (cache prep is inference-only and the
    # canonical [B, L_atom, L_atom, d] broadcast OOMs at ~14k+ atoms). Built-in fallback.
    use_chunked_inference = True
    # Atom-pair build-up instrumentation — opt-in dump of per-stage L2 norm
    # pooled to token-pair. The diffusion_module.py helpers handle the env
    # var check, counter cap, and NPZ write. Inference (this cache builder)
    # is the ONLY caller during normal runs — the in-place
    # _before_atom_transformer{,_chunked} methods on AtomAttentionEncoder
    # are bypassed by the cached diffusion step. So the dump must go here.
    from miniworld.modules.diffusion_module import (
        _next_call_idx, _pool_atom_pair_to_token, _should_dump, _dump_dir,
    )
    dump = _should_dump("cache_build")
    snap_pools: dict[str, "torch.Tensor"] = {}
    L_token_for_pool = int(scheme.token_idx.shape[1])
    if dump:
        snap_pools["0_init"] = _pool_atom_pair_to_token(
            atom_pair, atom_to_token_idx_map, L_token_for_pool,
        )

    # Dump path mirrors the exact ``add_`` sequence on atom_pair so each
    # captured stage is the cumulative state after one specific addition:
    #   0_init        : atom_pair = to_atom_pair(atom_pair_init)
    #   1_token_cond  : + _to_add_pair[b, A[b,i], A[b,j]]  (pair-cond gather)
    #   2_left        : + _left.unsqueeze(2)   (row-broadcast atom-single)
    #   3_right       : + _right.unsqueeze(1)  (col-broadcast atom-single)
    #   4_mlp         : + mlp_atom_pair(atom_pair)  (residual)
    # No stages are merged — every distinct ``add_`` gets its own row.
    if use_chunked_inference:
        _ATOM_CHUNK = 1024
        if dump:
            for s in range(0, atom_length, _ATOM_CHUNK):
                e = min(s + _ATOM_CHUNK, atom_length)
                row_tokens = atom_to_token_idx_map[:, s:e].unsqueeze(-1)
                atom_pair[:, s:e].add_(
                    _to_add_pair[b_arange, row_tokens, col_tokens]
                )
            snap_pools["1_token_cond"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
            for s in range(0, atom_length, _ATOM_CHUNK):
                e = min(s + _ATOM_CHUNK, atom_length)
                atom_pair[:, s:e].add_(_left[:, s:e].unsqueeze(2))
            snap_pools["2_left"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
            for s in range(0, atom_length, _ATOM_CHUNK):
                e = min(s + _ATOM_CHUNK, atom_length)
                atom_pair[:, s:e].add_(_right.unsqueeze(1))
            snap_pools["3_right"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
        else:
            for s in range(0, atom_length, _ATOM_CHUNK):
                e = min(s + _ATOM_CHUNK, atom_length)
                row_tokens = atom_to_token_idx_map[:, s:e].unsqueeze(-1)
                pair_slice = atom_pair[:, s:e]
                pair_slice.add_(
                    _to_add_pair[b_arange, row_tokens, col_tokens]
                )
                pair_slice.add_(_left[:, s:e].unsqueeze(2))
                pair_slice.add_(_right.unsqueeze(1))
        for s in range(0, atom_length, _ATOM_CHUNK):
            e = min(s + _ATOM_CHUNK, atom_length)
            atom_pair[:, s:e].add_(enc.mlp_atom_pair(atom_pair[:, s:e]))
    else:
        # Non-chunked: full pair gather is [B, L_atom, L_atom, d] — same
        # size as ``atom_pair`` itself, so peak doubles momentarily. Only
        # safe for small L_atom (kept as a reference; chunking is the default).
        row_tokens_full = atom_to_token_idx_map.unsqueeze(-1)  # [B, L_atom, 1]
        pair_cond = _to_add_pair[b_arange, row_tokens_full, col_tokens]
        if dump:
            atom_pair = atom_pair + pair_cond
            snap_pools["1_token_cond"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
            atom_pair = atom_pair + _left.unsqueeze(2)
            snap_pools["2_left"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
            atom_pair = atom_pair + _right.unsqueeze(1)
            snap_pools["3_right"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token_for_pool,
            )
        else:
            atom_pair = atom_pair + pair_cond + _left.unsqueeze(2) + _right.unsqueeze(1)
        atom_pair = atom_pair + enc.mlp_atom_pair(atom_pair)

    if dump:
        snap_pools["4_mlp"] = _pool_atom_pair_to_token(
            atom_pair, atom_to_token_idx_map, L_token_for_pool,
        )
        out_dir = _dump_dir()
        assert out_dir is not None
        call_idx = _next_call_idx("cache_build")
        import numpy as np
        np.savez(
            out_dir / f"atom_pair_cache_build_call{call_idx:04d}.npz",
            **{k: v.numpy() for k, v in snap_pools.items()},
            atom_to_token_idx_map=atom_to_token_idx_map[0].cpu().numpy(),
        )

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
        from team_gm.modules.layers import fourier_embedding
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
