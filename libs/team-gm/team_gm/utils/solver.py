import torch
from abc import ABC, abstractmethod
from typing import TypeVar, Any
from pydantic import BaseModel
from team_gm.utils.scheduler import DiffusionScheduler

schedulerT = TypeVar("T", bound="DiffusionScheduler")


class DiffusionSolver(ABC):
    """Base class for defining a diffusion solver."""

    class SolverSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

        # Add any additional configuration parameters here
        pass

    class SolverConfig(BaseModel):
        """Configuration for the DiffusionSolver class."""

        method: str = "Euler"
        seed: int = 0
        # Add any additional configuration parameters here
        pass

    def __init__(self, config: SolverConfig, scheduler: schedulerT):
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)

    def _set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ODEEulerSolver(DiffusionSolver):
    """
    A simple Euler ODE solver that uses the scheduler’s continuous‐time methods:

      - scheduler.sampling_time_steps(num_steps)
      - scheduler.sampling_schedule(time_steps)
      - scheduler.sampling_schedule_derivative(time_steps)
      - scheduler.sampling_scale(time_steps)
      - scheduler.sampling_scale_derivative(time_steps)
      - scheduler.output_scale(sigma)

    We assume:
      * model_fn(z, sigma) returns the model’s prediction of noise, i.e. εθ(z, sigma).
      * v_data = output_scale(sigma) * εθ is the “data‐domain velocity.”
      * dx/dt = α̇(t) · x  -  sigmȧ(t) · v_data.
    """

    def __init__(self, config: DiffusionSolver.SolverConfig, scheduler: schedulerT):
        super().__init__(config, scheduler)

    def step(
        self, model_fn: callable, x: torch.Tensor, t_index: int, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one Euler update in t-space.

        Args:
            model_fn: a callable `f(z, sigma)` → ε̂  (predicted noise at (normalized x, sigma))
            x: current sample, shape (B, C, H, W, …), in “noisy” (data) domain
            t_index: integer index in [0 .. len(time_steps)-2]
            time_steps: 1D tensor of time points, length = num_steps + 1

        Returns:
            x_{t_{i+1}} = x_{t_i} + Δt [ α̇(t_i) x_{t_i} - sigmȧ(t_i) · v_data(x_{t_i}, t_i ) ]
        """
        # 1. Get t_i and t_{i+1}, as well as Δt
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]
        dt = t_next - t_i  # scalar or tensor of shape (,)

        # 2. Query scheduler for sigma_i, α_i, and their derivatives
        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        # dsigma_dt_i = self.scheduler.sampling_schedule_derivative(t_i)  # sigmȧ(t_i)
        alpha_i = self.scheduler.sampling_scale(t_i)  # α(t_i)
        # dalpha_dt_i = self.scheduler.sampling_scale_derivative(t_i)  # α̇(t_i)

        # 3. Normalize x to “z” = x / α(t_i)
        #    We assume scheduler.input_scale(sigma) == α(sigma),
        #    so z_i = x / α_i.
        z_i = x / alpha_i

        # 4. Query the model for εθ(z_i, sigma_i)
        t_emb = self.scheduler.noise_condition(sigma_i)  # noise condition

        c_skip = self.scheduler.skip_scale(sigma_i)
        c_out = self.scheduler.output_scale(sigma_i)
        c_in = self.scheduler.input_scale(sigma_i)
        x_input = z_i * c_in  # normalized input to the model
        x_update = model_fn(x_input, t_emb)

        # for test
        x_update = torch.zeros_like(x_update)
        x_denoised = c_skip * x + c_out * x_update

        # 6. Compute dx/dt at t_i:  dx/dt = α̇(t_i) · x  -  sigmȧ(t_i) · v_data
        v_i = (x_denoised - x) / (sigma_i)

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        x_next = x - dt * v_i
        return x_next, x_update

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        A convenience wrapper that:
          1. Builds the time grid: t_0,...,t_N
          2. Initializes x_N as pure Gaussian noise ~ N(0, I)*sigma(t_0)
          3. Loops from i=0..(N-1), calling `step` each time.
          4. Returns x_0 (approximately denoised).

        Args:
            model_fn: f(z, sigma) → ε̂
            shape:   desired output shape (B, C, H, W, …)
            num_steps: how many discretization steps you want (N)
            device:  where to allocate tensors

        Returns:
            A tensor of shape `shape`, representing the decoded sample at t_N (≈0 noise).
        """
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        # e.g. shape = (num_steps + 1,)

        # 2. The initial noise level is at t_0
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])

        #    Draw x_N ~ N(0, I) * sigma_0
        x = torch.randn(shape, device=device) * sigma_0
        trajectory = []
        hat_list = []

        # 3. Iteratively step from i=0 to N-1
        for i in range(num_steps):
            x, epsilon_hat, what = self.step(model_fn, x, i, time_steps)
            if return_intermediate:
                trajectory.append(x.clone())
                hat_list.append(epsilon_hat.clone())

        # 4. Return x at t_N (typically sigma(t_N) ≈ 0, so x is “denoised”)
        if return_intermediate:
            return x, trajectory, hat_list
        else:
            return x


class SDESolver(DiffusionSolver):
    pass


class AF3Solver(DiffusionSolver):
    def __init__(self, config: DiffusionSolver.SolverConfig, scheduler: schedulerT):
        super().__init__(config, scheduler)

        # TODO: move it to config
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5

    def _set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    def step(
        self, model_fn: callable, x: torch.Tensor, t_index: int, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one Euler update in t-space.

        Args:
            model_fn: a callable `f(z, sigma)` → ε̂  (predicted noise at (normalized x, sigma))
            x: current sample, shape (B, C, H, W, …), in “noisy” (data) domain
            t_index: integer index in [0 .. len(time_steps)-2]
            time_steps: 1D tensor of time points, length = num_steps + 1

        Returns:
            x_{t_{i+1}} = x_{t_i} + Δt [ α̇(t_i) x_{t_i} - sigmȧ(t_i) · v_data(x_{t_i}, t_i ) ]
        """
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

        # x_update = torch.zeros_like(x_update)  # for test

        x_denoised = c_skip * x + c_out * x_update

        # 6. Compute dx/dt at t_i:  dx/dt = α̇(t_i) · x  -  sigmȧ(t_i) · v_data
        v_i = (x - x_denoised) / t_hat

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        x_next = x + self.step_scale * dt * v_i
        return x_next, x_update

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        A convenience wrapper that:
          1. Builds the time grid: t_0,...,t_N
          2. Initializes x_N as pure Gaussian noise ~ N(0, I)*sigma(t_0)
          3. Loops from i=0..(N-1), calling `step` each time.
          4. Returns x_0 (approximately denoised).

        Args:
            model_fn: f(z, sigma) → ε̂
            shape:   desired output shape (B, C, H, W, …)
            num_steps: how many discretization steps you want (N)
            device:  where to allocate tensors

        Returns:
            A tensor of shape `shape`, representing the decoded sample at t_N (≈0 noise).
        """
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        # e.g. shape = (num_steps + 1,)

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
        else:
            return x
