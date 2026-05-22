"""Per-step diffusion kernel that reuses the trunk-derived cache.

Inference equivalent of ``miniworld.modules.diffusion_module.DiffusionModule.forward``
that skips every computation already baked into the
:class:`~miniworld.inference.cache.InferenceCache`:

  - ``DiffusionConditioning`` pair branch (full ``token_pair_cond``)
  - ``DiffusionConditioning`` single branch (``token_single_cond[t]``
    precomputed for all T steps via the step schedule)
  - ``AtomAttentionEncoder._before_atom_transformer`` modulo the
    ``noisy_to_atom_single_rep(x_t)`` term (the only x_t-dependent piece)
  - ``_scatter_atom_to_token``'s one-hot mapping + count

The remaining per-step work is:

  1. ``atom_single_rep = cached atom_single_cond + noisy(x_t) * x_mask``
  2. encoder ``atom_transformer`` (atom_pair and atom_cond cached)
  3. scatter atoms -> tokens via cached mapping
  4. add cached ``add_single_token_cond[t]``
  5. token ``diffusion_transformer`` (cond + pair cached)
  6. atom_attention_decoder (cond + pair cached)

The function deliberately does **not** wear ``@typecheck``: this is a
tight loop and jaxtyping's runtime checks show up in flamegraphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from miniworld.inference.cache import InferenceCache, StepSchedule
    from miniworld.models.miniworld.model import Model


def diffusion_step(
    model: "Model",
    cache: "InferenceCache",
    schedule: "StepSchedule",
    x_t: torch.Tensor,
    t_index: int,
) -> torch.Tensor:
    """Run one diffusion-module forward for one solver step.

    Parameters
    ----------
    model
        Trained MiniWorld model. We reach into ``model.diffusion_module``
        to call the leaf nn.Modules with the cached inputs.
    cache
        Per-batch static cache (see :func:`build_inference_cache`).
    schedule
        Per-step precomputed schedule (see :func:`build_step_schedule`).
    x_t
        Noisy atom coordinates for this step, shape ``(A, L_atom, 3)``.
        ``A`` is the augmentation / sample axis.
    t_index
        Which timestep we are on. Indexes ``schedule.token_single_cond``
        and ``schedule.added_token_cond``.

    Returns
    -------
    Atom position update, shape ``(A, L_atom, 3)``.
    """
    dm = model.diffusion_module
    enc = dm.atom_attention_encoder
    dec = dm.atom_attention_decoder

    num_aug = x_t.shape[0]
    # Inject the model's B=1 axis: existing forward operates on (A, B, L_atom, 3).
    x_t_b = x_t.unsqueeze(1)
    atom_mask = cache.atom_mask                              # (B, L_atom) bool
    x_mask = atom_mask.unsqueeze(0).expand(num_aug, -1, -1)  # (A, B, L_atom)

    # ----- Atom encoder: only the noisy-x_t branch is recomputed -----
    atom_single_cond = cache.atom_single_cond_base           # (B, L_atom, d)
    atom_single_cond_aug = atom_single_cond.unsqueeze(0).expand(num_aug, -1, -1, -1)
    to_add = enc.noisy_to_atom_single_rep(x_t_b.to(torch.float32))  # (A, B, L_atom, d)
    to_add = to_add * x_mask.unsqueeze(-1)
    atom_single_rep = atom_single_cond.unsqueeze(0) + to_add

    atom_single_rep = enc.atom_transformer(
        atom_single_rep,
        atom_single_cond_aug,
        cache.atom_pair,                                     # (B, L_atom, L_atom, d_pair_atom)
        atom_mask,                                           # (B, L_atom)
    )

    # ----- Scatter atom -> token (cached mapping/count) -----
    am = atom_mask.unsqueeze(0).unsqueeze(-1)                # (1, B, L_atom, 1)
    atom_single_rep_masked = torch.where(
        am,
        atom_single_rep,
        torch.zeros_like(atom_single_rep),
    )
    to_add_single_token_rep = enc.atom_single_rep_to_token_single(atom_single_rep_masked)
    # canonical also masks after projection (defense in depth — see
    # _scatter_atom_to_token in diffusion_module.py).
    to_add_single_token_rep = to_add_single_token_rep * am
    token_single_rep = torch.einsum(
        "bal,nbac->nblc",
        cache.scatter_mapping,
        to_add_single_token_rep,
    ).contiguous()
    token_single_rep = token_single_rep * cache.scatter_count_inv.unsqueeze(0)
    # token_single_rep: (A, B, L_token, d_single_token)

    # ----- Token DiT (cond + pair cached, with leading-1 A broadcast) -----
    token_single_rep = token_single_rep + schedule.added_token_cond[t_index].unsqueeze(0)
    token_single_cond_t = schedule.token_single_cond[t_index].unsqueeze(0)  # (1, B, L_token, d_single)
    token_mask_aug = cache.token_mask.unsqueeze(0).expand(num_aug, -1, -1)
    token_single_rep = dm.diffusion_transformer(
        token_single_rep,
        token_single_cond_t,
        cache.token_pair_cond,
        mask=token_mask_aug,
    )
    token_single_rep = dm.ln_token_single_rep(token_single_rep)

    # ----- Atom decoder (atom_pair + atom_single_cond cached) -----
    # Mirror AtomAttentionDecoder.forward but feed it our cached tensors.
    atom_to_token_idx_map = cache.batch.scheme.atom_to_token_idx_map
    atom_length = atom_single_rep.shape[2]
    batch_size = atom_single_rep.shape[1]
    device = atom_single_rep.device

    batch_1d_idx = (
        torch.arange(batch_size, device=device)
        .view(1, batch_size, 1)
        .expand(num_aug, -1, atom_length)
    )
    aug_1d_idx = (
        torch.arange(num_aug, device=device)
        .view(num_aug, 1, 1)
        .expand(-1, batch_size, atom_length)
    )
    atom_to_token_idx_map_e = atom_to_token_idx_map.unsqueeze(0).expand(num_aug, -1, -1)

    _to_add_single = dec.add_token_info(token_single_rep)
    atom_single_rep = atom_single_rep + _to_add_single[
        aug_1d_idx, batch_1d_idx, atom_to_token_idx_map_e
    ]
    atom_single_rep = dec.atom_transformer(
        atom_single_rep,
        atom_single_cond_aug,
        cache.atom_pair,
        mask=atom_mask,
    )
    out = dec.final_denoising(atom_single_rep)               # (A, B, L_atom, 3)
    return out.squeeze(1)                                    # (A, L_atom, 3)
