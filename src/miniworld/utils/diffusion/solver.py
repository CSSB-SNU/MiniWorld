from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from miniworld.utils.diffusion.scheduler import (
    D3PMScheduler,
    DecoupledEDMScheduler,
    DiffusionScheduler,
    SEDDScheduler,
)
from miniworld.utils.structure.se3 import (
    apply_chain_rt,
    sample_rigid,
    se3_heat_step_delta_sigma,
    se3_heat_step_sigma,
)

SchedulerT = TypeVar("SchedulerT", bound="DiffusionScheduler")


def _symmetrize_pair(x: torch.Tensor) -> torch.Tensor:
    """Symmetrize pairwise tensors with channel as last dim (…, L, L, C)."""
    if x.ndim < 4:
        return x
    swapped = x.transpose(-3, -2)
    return 0.5 * (x + swapped)


def _symmetrize_labels(x: torch.Tensor) -> torch.Tensor:
    """Symmetrize integer labels for pairwise matrices."""
    if x.ndim < 3:
        return x
    if x.ndim == 3:
        return torch.triu(x) + torch.triu(x, 1).transpose(-1, -2)
    return x


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

    def __init__(self, config: SolverConfig, scheduler: SchedulerT) -> None:
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


class SEDDSolver(DiffusionSolver):
    """Solver for SEDD discrete diffusion using score-entropy ratios."""

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: SEDDScheduler,
        num_classes: int,
        enforce_symmetric: bool = True,
        min_ratio: float = 1e-5,
    ) -> None:
        super().__init__(config, scheduler)
        self.num_classes = num_classes
        self.enforce_symmetric = enforce_symmetric
        self.min_ratio = min_ratio

    def _base_init_labels(
        self,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        if self.scheduler.config.transition_mode == "absorbing":
            mask_class = self.scheduler.config.mask_class
            if mask_class is None:
                msg = "mask_class must be set for absorbing transition_mode."
                raise ValueError(msg)
            return torch.full(shape, mask_class, device=device, dtype=torch.long)
        # uniform base
        return torch.randint(
            low=0,
            high=self.num_classes,
            size=shape,
            device=device,
        )

    def step(
        self,
        model_fn: callable,
        xt_one_hot: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one ancestral sampling step."""
        scheduler: SEDDScheduler = self.scheduler
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]
        xt_labels = xt_one_hot.argmax(dim=-1)

        sigma_i = scheduler.sigma(t_i)
        t_emb = (
            scheduler.noise_condition(sigma_i).unsqueeze(0).repeat(xt_one_hot.shape[0])
        )
        ratio_pred = model_fn(xt_one_hot, t_emb).clamp_min(self.min_ratio)
        if self.enforce_symmetric:
            ratio_pred = _symmetrize_pair(ratio_pred)

        delta_sigma = scheduler.sigma_derivative(t_i, t_next).to(
            device=xt_one_hot.device,
            dtype=xt_one_hot.dtype,
        )
        q = scheduler.base_q(xt_labels, num_classes=self.num_classes)

        off_diag = 1.0 - xt_one_hot
        offdiag_trans = delta_sigma.unsqueeze(-1) * q * ratio_pred * off_diag
        offdiag_mass = offdiag_trans.sum(dim=-1, keepdim=True)
        diag_prob = (1.0 - offdiag_mass).clamp_min(0.0)
        trans = offdiag_trans + xt_one_hot * diag_prob
        trans = torch.clamp(trans, min=0.0)
        trans = trans / trans.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_prob = trans.view(trans.shape[0], -1, trans.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt_next = sampled.view(*xt_labels.shape)
        xt_one_hot_next = torch.nn.functional.one_hot(
            xt_next,
            num_classes=self.num_classes,
        ).to(xt_one_hot.dtype)

        if self.enforce_symmetric:
            xt_next = _symmetrize_labels(xt_next)
            xt_one_hot_next = torch.nn.functional.one_hot(
                xt_next,
                num_classes=self.num_classes,
            ).to(xt_one_hot.dtype)

        return xt_one_hot_next, ratio_pred

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Sample from the diffusion model using ancestral sampling."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)

        init_labels = self._base_init_labels(shape, device)
        xt_one_hot = torch.nn.functional.one_hot(
            init_labels,
            num_classes=self.num_classes,
        ).to(torch.float32)
        if self.enforce_symmetric:
            xt_one_hot = _symmetrize_pair(xt_one_hot)

        trajectory = []
        ratio_list = []
        for i in range(num_steps):
            xt_one_hot, ratio_pred = self.step(model_fn, xt_one_hot, i, time_steps)
            if return_intermediate:
                trajectory.append(xt_one_hot.clone())
                ratio_list.append(ratio_pred.clone())

        final_labels = xt_one_hot.argmax(dim=-1)
        if return_intermediate:
            return final_labels, trajectory, ratio_list
        return final_labels


class D3PMSolver(DiffusionSolver):
    """Ancestral sampler for D3PM discrete diffusion on pairwise labels."""

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: D3PMScheduler,
        num_classes: int,
        enforce_symmetric: bool = True,
    ) -> None:
        super().__init__(config, scheduler)
        self.num_classes = num_classes
        self.enforce_symmetric = enforce_symmetric

    def step(
        self,
        model_fn: callable,
        xt_one_hot: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one ancestral sampling step."""
        scheduler: D3PMScheduler = self.scheduler
        t = time_steps[t_index].long()
        xt_labels = xt_one_hot.argmax(dim=-1)

        t_emb = scheduler.noise_condition(t).unsqueeze(0).repeat(xt_one_hot.shape[0])
        logits_x0 = model_fn(xt_one_hot, t_emb)
        x0_hat = logits_x0.argmax(dim=-1)

        q_post = scheduler.q_posterior(
            xt_labels,
            x0_hat,
            t,
            num_classes=self.num_classes,
        )

        flat_prob = q_post.view(q_post.shape[0], -1, q_post.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt_minus_1 = sampled.view(*x0_hat.shape)
        xt_minus_1_one_hot = F.one_hot(
            xt_minus_1,
            num_classes=self.num_classes,
        ).to(dtype=xt_one_hot.dtype)

        if self.enforce_symmetric:
            xt_minus_1 = _symmetrize_labels(xt_minus_1)
            xt_minus_1_one_hot = F.one_hot(
                xt_minus_1,
                num_classes=self.num_classes,
            ).to(dtype=xt_one_hot.dtype)

        return xt_minus_1_one_hot, logits_x0

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Sample from the diffusion model using ancestral sampling."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        if self.scheduler.config.transition_mode == "absorbing":
            mask_class = self.scheduler.config.mask_class
            if mask_class is None:
                msg = "mask_class must be set for absorbing transition_mode."
                raise ValueError(msg)
            labels = torch.full(shape, mask_class, device=device, dtype=torch.long)
        else:
            labels = torch.randint(
                low=0,
                high=self.num_classes,
                size=shape,
                device=device,
            )
        xt = F.one_hot(labels, num_classes=self.num_classes).to(torch.float32)
        trajectory = []
        logits_list = []
        for i in range(num_steps):
            xt, logits_x0 = self.step(model_fn, xt, i, time_steps)
            if return_intermediate:
                trajectory.append(xt.clone())
                logits_list.append(logits_x0.clone())

        final_labels = xt.argmax(dim=-1)
        if return_intermediate:
            return final_labels, trajectory, logits_list
        return final_labels


class ODEEulerSolver(DiffusionSolver):
    """A simple Euler ODE solver that uses the scheduler's continuous-time methods below.

      - scheduler.sampling_time_steps(num_steps)
      - scheduler.sampling_schedule(time_steps)
      - scheduler.sampling_schedule_derivative(time_steps)
      - scheduler.sampling_scale(time_steps)
      - scheduler.sampling_scale_derivative(time_steps)
      - scheduler.output_scale(sigma)

    We assume:
      * model_fn(z, sigma) returns the model's prediction of noise, i.e. εθ(z, sigma).
      * v_data = output_scale(sigma) * εθ is the “data-domain velocity.”
      * dx/dt = α̇(t) · x  -  sigmȧ(t) · v_data.
    """  # noqa: RUF002

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: SchedulerT,
    ) -> None:
        super().__init__(config, scheduler)

    def step(
        self,
        model_fn: callable,
        x: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Euler update in t-space.

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
        alpha_i = self.scheduler.sampling_scale(t_i)  # α(t_i)

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
        """Sample from the diffusion model using the ODE Euler solver."""
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
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
        return x


class SDESolver(DiffusionSolver):
    """A base class for SDE-based diffusion solvers."""


class AF3Solver(DiffusionSolver):
    """A solver implementing the AF3 method."""

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: SchedulerT,
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
        model_fn: callable,
        x: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Euler update in t-space."""
        # 1. Get t_i and t_{i+1}, as well as Δt
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)

        added_noise = (
            self._lambda * (t_hat**2 - sigma_i**2) ** 0.5 * torch.randn_like(x)
        )

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

        # 6. Compute dx/dt at t_i:  dx/dt = α̇(t_i) · x  -  sigmȧ(t_i) · v_data
        v_i = (x - x_denoised) / t_hat

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        x_next = x + self.step_scale * dt * v_i
        return x_next, x_denoised

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
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
    """A solver implementing the Decoupled EDM method."""

    def __init__(
        self,
        config: DiffusionSolver.SolverConfig,
        scheduler: DecoupledEDMScheduler,
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

    def _add_noise(
        self,
        y: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        sigma_Ri, sigma_Ti = self.scheduler.convert_to_sigmaRT(sigma_i)

        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        sigma_Rhat, sigma_That = self.scheduler.convert_to_sigmaRT(t_hat)
        R_hat, T_hat = se3_heat_step_sigma(
            R,
            T,
            sigma_Ri,
            sigma_Ti,
            sigma_Rhat,
            sigma_That,
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
        model_fn: callable,
        y: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: dict,
    ) -> torch.Tensor:
        """Perform one Euler update in t-space."""
        # 1. Get t_i and t_{i+1}, as well as Δt
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        _, sigma_That = self.scheduler.convert_to_sigmaRT(t_hat)

        # add noise
        y, x_with_noise, t_hat = self._add_noise(
            y,
            R,
            T,
            t_index,
            time_steps,
            atom_chain_break,
        )
        dt = sigma_next - t_hat

        # 4. Query the model for εθ(z_i, sigma_i)
        t_emb = self.scheduler.noise_condition(t_hat)  # noise condition
        c_skip = self.scheduler.skip_scale(t_hat)
        c_out = self.scheduler.output_scale(t_hat)
        c_in = self.scheduler.input_scale(t_hat, sigma_That)
        x_input = x_with_noise * c_in  # normalized input to the model
        x_update = model_fn(x_input, t_emb)

        x_denoised = c_skip * y + c_out * x_update

        # 6. Compute dx/dt at t_i:  dx/dt = α̇(t_i) · x  -  sigmȧ(t_i) · v_data
        v_i = (y - x_denoised) / t_hat

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        y = y + self.step_scale * dt * v_i

        return y, x_update

    def RT_step(  # noqa: N802
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one SE3 update step."""
        # 1. Get t_i and t_{i+1}
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        # 2. Compute the time step
        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        sigma_Ri, sigma_Ti = self.scheduler.convert_to_sigmaRT(sigma_i)
        sigma_Rnext, sigma_Tnext = self.scheduler.convert_to_sigmaRT(sigma_next)
        dt_R, dt_T = sigma_Rnext - sigma_Ri, sigma_Tnext - sigma_Ti

        R_i, T_i = se3_heat_step_delta_sigma(R, T, sigma_Ri, sigma_Ti, dt_R, dt_T)

        return R_i, T_i

    def step(
        self,
        model_fn: callable,
        y: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
        atom_chain_break: dict,
    ) -> torch.Tensor:
        """Perform one Euler update in t-space."""
        y, x_update = self.y_step(
            model_fn,
            y,
            R,
            T,
            t_index,
            time_steps,
            atom_chain_break,
        )
        R, T = self.RT_step(R, T, t_index, time_steps)
        return y, x_update, R, T

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        atom_chain_break: dict,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Sample from the diffusion model using the ODE Euler solver."""
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)

        # 2. The initial noise level is at t_0
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])
        sigma_R, sigma_T = self.scheduler.convert_to_sigmaRT(time_steps[0])

        B = shape[0]
        sigma_R = sigma_R.expand(B)
        sigma_T = sigma_T.expand(B)

        #    Draw x_N ~ N(0, I) * sigma_0
        y = torch.randn(shape, device=device) * sigma_0
        chain_num = len(atom_chain_break)
        R, T = sample_rigid(sigma_R, sigma_T, C=chain_num, device=device)

        trajectory = []
        hat_list = []

        # 3. Iteratively step from i=0 to N-1
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

        # 4. Return y at t_N (typically sigma(t_N) ≈ 0, so y is “denoised”)
        if return_intermediate:
            return y, trajectory, hat_list
        return y


class SPELLSolver(DiffusionSolver):
    """A solver implementing the SPELL method."""

    class SolverConfig(BaseModel):
        """Configuration for the DiffusionSolver class."""

        method: str = "Euler"
        seed: int = 0
        radius: float = 5.0
        spell_lambda: float = 1.6

    def __init__(self, config: SolverConfig, scheduler: SchedulerT) -> None:
        super().__init__(config, scheduler)

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5
        self.radius = config.radius
        self.spell_lambda = config.spell_lambda

        # SPELL specific
        self.x0_list: list[torch.Tensor] = []

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
        model_fn: callable,
        x: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Euler update in t-space and apply SPELL.

        Args:
            model_fn: a callable `f(z, sigma)` → ε̂  (predicted noise at (normalized x, sigma))
            x: current sample, shape, in “noisy” (data) domain
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

        added_noise = (
            self._lambda * (t_hat**2 - sigma_i**2) ** 0.5 * torch.randn_like(x)
        )

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

        if len(self.x0_list) == 0:
            spell_force = torch.zeros_like(x)
        else:
            ref_x0s = torch.stack(self.x0_list, dim=0)  # (R, B, L, 3)
            diff = x_denoised.unsqueeze(0) - ref_x0s  # [R, ...]
            diff = diff.view(diff.shape[0], -1)  # [R, D]
            diff_norm = diff.norm(dim=-1)  # [R]
            act = torch.nn.functional.relu(self.radius / diff_norm - 1)  # [R]
            delta = (act.unsqueeze(-1) * diff).sum(dim=0)
            spell_force = delta.view_as(x_denoised)

        x_denoised = x_denoised + self.spell_lambda * spell_force

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

        # append x0
        self.x0_list.append(x)

        # 4. Return x at t_N (typically sigma(t_N) ≈ 0, so x is “denoised”)
        if return_intermediate:
            return x, trajectory, hat_list
        return x
