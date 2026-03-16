from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Float
from pydantic import BaseModel

from miniworld.diffusion.scheduler import (
    DiffusionScheduler,
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
        time_steps: Float[torch.Tensor, "..."],
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
