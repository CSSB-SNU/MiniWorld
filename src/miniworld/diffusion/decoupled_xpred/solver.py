"""VE x-prediction ODE solver (Decoupled coordinate + rigid-body SE(3)).

R/T are auxiliary model-input corruptions sampled fresh at each step from
their marginal at sigma_hat — they carry no state across steps, so they do
not appear in the public step/sample interface.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel

from miniworld.diffusion.base.solver import (
    DiffusionSolver,
    ModelFn,
    _chain_count,
    _expand_to_batch,
)
from miniworld.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler
from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid


class XPredDecoupledSolver(DiffusionSolver):
    """EDM ODE solver with x-prediction for decoupled coordinate + R/T noise.

    The model outputs x0/sigma_data. The solver multiplies by sigma_data
    to recover x_pred in original coordinates for the ODE step.
    """

    class Config(BaseModel):
        """Configuration for XPredDecoupledSolver."""

        seed: int = 0
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_lambda: float = 1.003
        step_scale: float = 1.5

    def __init__(
        self,
        config: Config,
        scheduler: DecoupledXPredScheduler,
    ) -> None:
        self.config = config
        self.scheduler: DecoupledXPredScheduler = scheduler
        self._set_seed(config.seed)

    @property
    def sigma_data(self) -> float:
        """Data standard deviation from scheduler config."""
        return self.scheduler.config.sigma_data

    def _sample_rt(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        chain_num: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample R/T marginal at given sigma."""
        sigma_r, sigma_t = self.scheduler.convert_to_sigma_rt(sigma)
        sigma_r = _expand_to_batch(sigma_r, batch_size)
        sigma_t = _expand_to_batch(sigma_t, batch_size)
        return sample_rigid(
            sigma_r,
            sigma_t,
            C=chain_num,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        model_fn: ModelFn,
        y: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_to_combine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One solver step on coordinates.

        R/T are sampled fresh from their marginal at sigma_hat (not stateful).
        Returns (y_next, x_pred).
        """
        sigma_i = self.scheduler.sampling_schedule(time_steps[t_index])
        sigma_next = self.scheduler.sampling_schedule(time_steps[t_index + 1])
        batch_size = y.shape[0]
        chain_num = _chain_count(atom_to_combine)

        # Stochastic noise injection (EDM Euler)
        gamma = self.config.gamma_0 if sigma_next > self.config.gamma_min else 0
        sigma_hat = sigma_i * (1 + gamma)
        if gamma > 0:
            added_noise = (
                self.config.noise_lambda
                * (sigma_hat**2 - sigma_i**2) ** 0.5
                * torch.randn_like(y)
            )
            y = y + added_noise

        # Sample R/T marginal at sigma_hat for model-input corruption
        R_hat, T_hat = self._sample_rt(
            sigma_hat,
            batch_size,
            chain_num,
            y.device,
            y.dtype,
        )
        x_with_noise = apply_chain_rt(y, R_hat, T_hat, atom_to_combine)

        # Model prediction
        _, sigma_t_hat = self.scheduler.convert_to_sigma_rt(sigma_hat)
        t_emb = self.scheduler.noise_condition(sigma_hat)
        c_in = self.scheduler.input_scale(sigma_hat, sigma_t_hat)
        x_pred = model_fn(x_with_noise * c_in, t_emb) * self.sigma_data

        # EDM ODE step: dy/dsigma = (y - x_pred) / sigma
        v = (y - x_pred) / sigma_hat
        y_next = y + self.config.step_scale * (sigma_next - sigma_hat) * v

        return y_next, x_pred

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        shape: torch.Size,
        atom_to_combine: torch.Tensor,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
        """Sample from noise using EDM ODE with x-prediction."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])

        y = torch.randn(shape, device=device) * sigma_0

        trajectory: list[torch.Tensor] = []
        hat_list: list[torch.Tensor] = []

        for i in range(num_steps):
            y, x_pred = self.step(
                model_fn,
                y,
                i,
                time_steps,
                atom_to_combine,
            )
            if return_intermediate:
                trajectory.append(y.clone())
                hat_list.append(x_pred.clone())

        if return_intermediate:
            return y, trajectory, hat_list
        return y
