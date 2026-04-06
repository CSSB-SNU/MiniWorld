from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
from jaxtyping import Float
from pydantic import BaseModel

from miniworld.diffusion.scheduler import DecoupledEDMScheduler, DiffusionScheduler
from miniworld.utils.structure.se3 import (
    apply_chain_rt,
    sample_rigid,
    se3_heat_step_delta_sigma,
    se3_heat_step_sigma,
)


class DiffusionSolver(ABC):
    """Base class for defining a diffusion solver."""

    class SolverSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

        # Add any additional configuration parameters here

    class SolverConfig(BaseModel):
        """Configuration for the DiffusionSolver class."""

        method: str = "Euler"
        seed: int = 0
        # Add any additional configuration parameters here

    def __init__(self, config: SolverConfig, scheduler: DiffusionScheduler) -> None:
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform one solver step."""


ModelFn = Callable[
    [Float[torch.Tensor, "... L 3"], Float[torch.Tensor, "..."]],
    Float[torch.Tensor, "... L 3"],
]


AtomChainMap = torch.Tensor | Mapping[Any, tuple[int, int]]


def _expand_to_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a 1D tensor with one value per batch item."""
    value = value.reshape(-1)
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        msg = f"Expected scalar or {batch_size} values, got shape {value.shape}."
        raise ValueError(msg)
    return value


def _chain_count(atom_chain_map: AtomChainMap) -> int:
    if isinstance(atom_chain_map, torch.Tensor):
        if atom_chain_map.numel() == 0:
            msg = "atom_chain_map must not be empty."
            raise ValueError(msg)
        return int(atom_chain_map.max().item()) + 1
    return len(atom_chain_map)


class AF3Solver(DiffusionSolver):
    """A solver implementing the AF3 method."""

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        super().__init__(config, scheduler)
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    def step(
        self,
        model_fn: ModelFn,
        x: Float[torch.Tensor, "... L 3"],
        t_index: int,
        time_steps: Float[torch.Tensor, ...],
    ) -> tuple[Float[torch.Tensor, "... L 3"], Float[torch.Tensor, "... L 3"]]:
        """Perform one Euler update in t-space."""
        # 1. Get t_i and t_{i+1}, as well as Δt
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)

        added_noise = self._lambda * (t_hat**2 - sigma_i**2) ** 0.5 * torch.randn_like(x)

        x = x + added_noise

        dt = sigma_next - t_hat

        # 4. Query the model for εθ(z_i, sigma_i)
        t_emb = self.scheduler.noise_condition(t_hat)  # noise condition
        c_skip = self.scheduler.skip_scale(t_hat)
        c_out = self.scheduler.output_scale(t_hat)
        c_in = self.scheduler.input_scale(t_hat)
        x_input = x * c_in  # normalized input to the model
        x_update = model_fn(x_input, t_emb)

        x_denoised = c_skip * x + c_out * x_update

        # 6. Compute dx/dt at t_i
        v_i = (x - x_denoised) / t_hat

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        x_next = x + self.step_scale * dt * v_i
        return x_next, x_denoised

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
        """Sample from the diffusion model using the ODE Euler solver."""
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)

        # 2. The initial noise level is at t_0
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])

        #    Draw x_N ~ N(0, I) * sigma_0
        x = -1 * torch.randn(shape, device=device) * sigma_0
        trajectory = []
        hat_list = []

        # 3. Iteratively step from i=0 to N-1
        for i in range(num_steps):
            x, epsilon_hat = self.step(model_fn, x, i, time_steps)
            if return_intermediate:
                trajectory.append(x.clone())
                hat_list.append(epsilon_hat.clone())

        # 4. Return x at t_N (typically sigma(t_N) ≈ 0, so x is “denoised”)
        if return_intermediate:
            return x, trajectory, hat_list
        return x


class DecoupledEDMSolver(DiffusionSolver):
    """A solver implementing the decoupled EDM method."""

    scheduler: DecoupledEDMScheduler

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: DecoupledEDMScheduler,
    ) -> None:
        super().__init__(config, scheduler)
        self.scheduler: DecoupledEDMScheduler = scheduler
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    def _add_noise(
        self,
        y: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: AtomChainMap,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add coordinate and SE(3) noise for the current solver step."""
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)
        sigma_rotation_i, sigma_translation_i = self.scheduler.convert_to_sigma_rt(
            sigma_i,
        )
        batch_size = y.shape[0]
        sigma_rotation_i = _expand_to_batch(sigma_rotation_i, batch_size)
        sigma_translation_i = _expand_to_batch(sigma_translation_i, batch_size)

        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        sigma_rotation_hat, sigma_translation_hat = self.scheduler.convert_to_sigma_rt(
            t_hat,
        )
        sigma_rotation_hat = _expand_to_batch(sigma_rotation_hat, batch_size)
        sigma_translation_hat = _expand_to_batch(sigma_translation_hat, batch_size)
        R_hat, T_hat = se3_heat_step_sigma(
            rotation,
            translation,
            sigma_rotation_i,
            sigma_translation_i,
            sigma_rotation_hat,
            sigma_translation_hat,
            eps=1e-12,
        )

        added_noise = (
            self._lambda * (t_hat**2 - sigma_i**2) ** 0.5 * torch.randn_like(y)
        )
        y = y + added_noise
        x_with_noise = apply_chain_rt(y, R_hat, T_hat, atom_chain_break)
        return y, x_with_noise, t_hat

    def y_step(
        self,
        model_fn: ModelFn,
        y: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: AtomChainMap,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one Euler update on the coordinate component."""
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)
        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        _, sigma_translation_hat = self.scheduler.convert_to_sigma_rt(t_hat)

        y, x_with_noise, t_hat = self._add_noise(
            y,
            rotation,
            translation,
            t_index,
            time_steps,
            atom_chain_break,
        )
        dt = sigma_next - t_hat

        t_emb = self.scheduler.noise_condition(t_hat)
        c_skip = self.scheduler.skip_scale(t_hat)
        c_out = self.scheduler.output_scale(t_hat)
        c_in = self.scheduler.input_scale(t_hat, sigma_translation_hat)
        x_input = x_with_noise * c_in
        x_update = model_fn(x_input, t_emb)

        x_denoised = c_skip * y + c_out * x_update
        v_i = (y - x_denoised) / t_hat
        y = y + self.step_scale * dt * v_i

        return y, x_update

    def rt_step(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one SE(3) update step."""
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)
        sigma_rotation_i, sigma_translation_i = self.scheduler.convert_to_sigma_rt(
            sigma_i,
        )
        batch_size = rotation.shape[0]
        sigma_rotation_i = _expand_to_batch(sigma_rotation_i, batch_size)
        sigma_translation_i = _expand_to_batch(sigma_translation_i, batch_size)
        sigma_rotation_next, sigma_translation_next = (
            self.scheduler.convert_to_sigma_rt(sigma_next)
        )
        sigma_rotation_next = _expand_to_batch(sigma_rotation_next, batch_size)
        sigma_translation_next = _expand_to_batch(sigma_translation_next, batch_size)
        dt_rotation = sigma_rotation_next - sigma_rotation_i
        dt_translation = sigma_translation_next - sigma_translation_i

        return se3_heat_step_delta_sigma(
            rotation,
            translation,
            sigma_rotation_i,
            sigma_translation_i,
            dt_rotation,
            dt_translation,
        )

    def step(
        self,
        model_fn: ModelFn,
        y: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: AtomChainMap,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one solver step for coordinates and rigid transforms."""
        y, x_update = self.y_step(
            model_fn,
            y,
            rotation,
            translation,
            t_index,
            time_steps,
            atom_chain_break,
        )
        rotation, translation = self.rt_step(
            rotation,
            translation,
            t_index,
            time_steps,
        )
        return y, x_update, rotation, translation

    def sample(
        self,
        model_fn: ModelFn,
        shape: torch.Size,
        atom_chain_break: AtomChainMap,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> (
        tuple[
            torch.Tensor,
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | torch.Tensor
    ):
        """Sample from the diffusion model using the decoupled Euler solver."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)

        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])
        sigma_rotation, sigma_translation = self.scheduler.convert_to_sigma_rt(
            time_steps[0],
        )

        batch_size = shape[0]
        sigma_rotation = sigma_rotation.expand(batch_size)
        sigma_translation = sigma_translation.expand(batch_size)

        y = torch.randn(shape, device=device) * sigma_0
        chain_num = _chain_count(atom_chain_break)
        R, T = sample_rigid(
            sigma_rotation,
            sigma_translation,
            C=chain_num,
            device=device,
            dtype=y.dtype,
        )

        trajectory = []
        hat_list = []

        for i in range(num_steps):
            y, epsilon_hat, R, T = self.step(
                model_fn,
                y,
                R,
                T,
                i,
                time_steps,
                atom_chain_break,
            )
            if return_intermediate:
                trajectory.append(y.clone())
                hat_list.append(epsilon_hat.clone())

        if return_intermediate:
            return y, trajectory, hat_list
        return y
