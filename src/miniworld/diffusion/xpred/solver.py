"""VE x-prediction ODE solver (Euclidean).

Uses the standard EDM ODE:  dy/dsigma = (y - x_pred) / sigma
with x-prediction:          F_theta(c_in * y; sigma) = x0 / sigma_data
                            x_pred = F_theta * sigma_data
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from pydantic import BaseModel

from miniworld.diffusion.base.solver import DiffusionSolver, ModelFn
from miniworld.diffusion.edm.scheduler import EDMScheduler


class XPredEulerSolver(DiffusionSolver):
    """EDM ODE solver with x-prediction for Euclidean coordinates.

    The model outputs x0/sigma_data. The solver multiplies by sigma_data
    to recover x_pred in original coordinates for the ODE step.

    Interface is identical to ``AF3Solver``.
    """

    class Config(BaseModel):
        """Configuration for XPredEulerSolver."""

        seed: int = 0
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_lambda: float = 1.003
        step_scale: float = 1.5

    def __init__(self, config: Config, scheduler: EDMScheduler) -> None:
        self.config = config
        self.scheduler: EDMScheduler = scheduler
        self._set_seed(config.seed)

    @property
    def sigma_data(self) -> float:
        """Data standard deviation from scheduler config."""
        return self.scheduler.config.sigma_data

    def step(
        self,
        model_fn: ModelFn,
        x: Float[torch.Tensor, "... L 3"],
        t_index: int,
        time_steps: Float[torch.Tensor, ...],
    ) -> tuple[Float[torch.Tensor, "... L 3"], Float[torch.Tensor, "... L 3"]]:
        """One Euler step with optional stochastic noise."""
        sigma_i = self.scheduler.sampling_schedule(time_steps[t_index])
        sigma_next = self.scheduler.sampling_schedule(time_steps[t_index + 1])

        gamma = self.config.gamma_0 if sigma_next > self.config.gamma_min else 0
        sigma_hat = sigma_i * (1 + gamma)
        if gamma > 0:
            added_noise = (
                self.config.noise_lambda
                * (sigma_hat**2 - sigma_i**2) ** 0.5
                * torch.randn_like(x)
            )
            x = x + added_noise

        dt = sigma_next - sigma_hat

        t_emb = self.scheduler.noise_condition(sigma_hat)
        c_in = self.scheduler.input_scale(sigma_hat)
        x_pred = model_fn(x * c_in, t_emb) * self.sigma_data

        v = (x - x_pred) / sigma_hat
        x_next = x + self.config.step_scale * dt * v
        return x_next, x_pred

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        *,
        return_intermediate: bool = False,
    ) -> (
        tuple[
            Float[torch.Tensor, "... L 3"],
            list[Float[torch.Tensor, "... L 3"]],
            list[Float[torch.Tensor, "... L 3"]],
        ]
        | Float[torch.Tensor, "... L 3"]
    ):
        """Sample from noise using EDM ODE with x-prediction."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])

        x = torch.randn(shape, device=device) * sigma_0
        trajectory: list[torch.Tensor] = []
        hat_list: list[torch.Tensor] = []

        for i in range(num_steps):
            x, x_pred = self.step(model_fn, x, i, time_steps)
            if return_intermediate:
                trajectory.append(x.clone())
                hat_list.append(x_pred.clone())

        if return_intermediate:
            return x, trajectory, hat_list
        return x
