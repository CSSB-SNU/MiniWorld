"""Slim inference solver.

Same math as :class:`miniworld.diffusion.decoupled_xpred.solver.XPredDecoupledSolver`,
but:

  - Sigmas, c_in, gamma etc. are looked up from the prebuilt
    :class:`~miniworld.inference.cache.StepSchedule` instead of being
    recomputed each step.
  - The model call goes through :func:`diffusion_step` (cache-aware) and
    takes ``t_index`` directly — no scalar ``t_emb`` needs to be
    threaded.
  - Nothing here runs ``@typecheck`` or holds gradient tape.

R/T noise is still sampled per-step (it carries no state across steps
but does require fresh randomness — that's the whole point of the
auxiliary corruption).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from miniworld.diffusion.base.solver import _chain_count, _expand_to_batch
from miniworld.inference.diffusion import diffusion_step
from miniworld.inference.ligand_potential import LigandRestraint, apply_ligand_restraint
from miniworld.utils.structure.align import weighted_align
from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid

if TYPE_CHECKING:
    from miniworld.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler
    from miniworld.diffusion.decoupled_xpred.solver import XPredDecoupledSolver
    from miniworld.inference.cache import InferenceCache, StepSchedule
    from miniworld.models.miniworld.model import Model


def _prepare_weight(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
    weight = mask.to(device=x.device, dtype=x.dtype)
    if weight.ndim == 1:
        weight = weight.unsqueeze(0).expand(x.shape[0], -1)
    elif weight.shape[0] == 1 and x.shape[0] > 1:
        weight = weight.expand(x.shape[0], -1)
    return weight


def _center_to_origin(
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return x - x.mean(dim=-2, keepdim=True)
    weight = _prepare_weight(x, mask)
    denom = weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
    centroid = (x * weight.unsqueeze(-1)).sum(dim=-2, keepdim=True) / denom.unsqueeze(-1)
    return (x - centroid) * weight.unsqueeze(-1)


def _align_to_prediction(
    y: torch.Tensor,
    x_pred: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = _prepare_weight(y, mask)
    y_aligned = weighted_align(y, x_pred, weight=weight)
    return torch.where(weight.unsqueeze(-1) > 0, y_aligned, y)


def _sample_rt(
    scheduler: "DecoupledXPredScheduler",
    sigma_hat: torch.Tensor,
    sigma_t_hat: torch.Tensor,
    batch_size: int,
    chain_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_r, _ = scheduler.convert_to_sigma_rt(sigma_hat)
    sigma_r = _expand_to_batch(sigma_r, batch_size)
    sigma_t = _expand_to_batch(sigma_t_hat, batch_size)
    return sample_rigid(sigma_r, sigma_t, C=chain_num, device=device, dtype=dtype)


@torch.inference_mode()
def sample_trajectory(
    model: "Model",
    cache: "InferenceCache",
    schedule: "StepSchedule",
    solver_cfg: "XPredDecoupledSolver.Config",
    scheduler: "DecoupledXPredScheduler",
    *,
    n_samples: int,
    atom_to_combine: torch.Tensor,
    device: torch.device,
    use_rt: bool = True,
    update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
    init_x0: torch.Tensor | None = None,
    return_intermediate: bool = True,
    ligand_restraint: "LigandRestraint | None" = None,
    ligand_sigma_threshold: float | None = None,
    ligand_steps: int = 20,
    ligand_lr: float = 0.05,
    ligand_w_tether: float = 0.1,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Run the diffusion solver. Returns ``(y_final, inter, x0_hat, x_input)``.

    Equivalent (up to RNG ordering) to
    ``XPredDecoupledSolver.sample(... return_intermediate=True)`` but
    consults :class:`StepSchedule` for sigmas and goes through
    :func:`diffusion_step` for the model call.
    """
    num_steps = int(schedule.sigma_hat.shape[0])
    n_atoms = int(cache.batch.structure.atom_pos.shape[1])

    # y_0
    sigma_0 = schedule.time_steps[0]
    if init_x0 is not None:
        init_x0 = init_x0.to(device=device, dtype=torch.float32)
        if init_x0.ndim == 2:
            init_x0 = init_x0.unsqueeze(0)
        shape = (n_samples, n_atoms, 3)
        if init_x0.shape[-2:] != shape[-2:]:
            msg = (
                f"init_x0 shape {tuple(init_x0.shape)} does not match "
                f"sample shape {shape}; last two dims must agree."
            )
            raise ValueError(msg)
        init_x0 = init_x0.expand(shape)
        y = init_x0 + sigma_0 * torch.randn(shape, device=device)
    else:
        y = torch.randn((n_samples, n_atoms, 3), device=device) * sigma_0

    atom_pos_mask = cache.batch.structure.atom_pos_mask.bool()
    mask_for_solver = atom_pos_mask if atom_pos_mask.shape[0] == n_samples else atom_pos_mask
    # The existing solver passes ``batch.structure.atom_pos_mask.bool()`` as the
    # weighted-align mask; (1, L_atom) broadcasts to (n_samples, L_atom) in
    # ``_prepare_weight``. No special handling needed.

    chain_num = _chain_count(atom_to_combine)
    sigma_data = scheduler.config.sigma_data
    noise_lambda = solver_cfg.noise_lambda
    step_scale = solver_cfg.step_scale

    # Ligand steering only fires in the low-noise regime where x0_hat is a
    # trustworthy clean-structure estimate (and where geometry crystallises /
    # bonds break). Default threshold = sigma_data. See ligand_potential.py.
    if ligand_sigma_threshold is None:
        ligand_sigma_threshold = float(sigma_data)

    inter_traj: list[torch.Tensor] = []
    hat_list: list[torch.Tensor] = []
    input_list: list[torch.Tensor] = []

    for t_index in range(num_steps):
        sigma_i = schedule.sigma_i[t_index]
        sigma_hat = schedule.sigma_hat[t_index]
        sigma_next = schedule.sigma_next[t_index]
        sigma_t_hat = schedule.sigma_t_hat[t_index]
        c_in = schedule.c_in[t_index]
        gamma = schedule.gamma[t_index]
        noise_scale = schedule.noise_scale[t_index]

        y_pre_inject = y
        if float(gamma) > 0:
            added = (
                noise_lambda
                * (sigma_hat ** 2 - sigma_i ** 2).clamp(min=0.0).sqrt()
                * torch.randn_like(y)
            )
            y = y + added

        if use_rt:
            rotation_hat, translation_hat = _sample_rt(
                scheduler,
                sigma_hat,
                sigma_t_hat,
                batch_size=n_samples,
                chain_num=chain_num,
                device=device,
                dtype=y.dtype,
            )
            x_with_noise = apply_chain_rt(
                y,
                rotation_hat,
                translation_hat,
                atom_to_combine,
            )
        else:
            x_with_noise = y

        x_input = x_with_noise * c_in
        x_pred = diffusion_step(
            model, cache, schedule, x_input, t_index,
        ) * sigma_data

        # Ligand geometry steering: project the clean-structure estimate toward
        # valid bond/angle geometry (torsions + pose left free). Applied to
        # x_pred so it flows into whichever update rule follows; gated to the
        # low-noise regime. No-op when no ligand restraint was built.
        if ligand_restraint is not None and float(sigma_hat) < ligand_sigma_threshold:
            x_pred = apply_ligand_restraint(
                x_pred,
                ligand_restraint,
                n_steps=ligand_steps,
                lr=ligand_lr,
                w_tether=ligand_w_tether,
            )

        if update_rule == "ode":
            v = (y_pre_inject - x_pred) / sigma_hat
            y = y + step_scale * (sigma_next - sigma_hat) * v
        elif update_rule == "ode_aligned":
            y_aligned = _align_to_prediction(y, x_pred, mask_for_solver)
            v = (y_aligned - x_pred) / sigma_hat
            y = y_aligned + step_scale * (sigma_next - sigma_hat) * v
        elif update_rule == "x0_centered":
            x_pred = _center_to_origin(x_pred, mask_for_solver)
            is_last_step = t_index == num_steps - 1
            if is_last_step:
                y = x_pred
            else:
                y = x_pred + noise_scale * torch.randn_like(y)
        else:
            msg = f"Unsupported update_rule: {update_rule}"
            raise ValueError(msg)

        if return_intermediate:
            inter_traj.append(y.clone())
            hat_list.append(x_pred.clone())
            input_list.append(x_with_noise.clone())

    return y, inter_traj, hat_list, input_list
