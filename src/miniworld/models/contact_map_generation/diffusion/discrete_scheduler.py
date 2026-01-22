
from math import pi
from typing import Literal

import torch
from team_gm.utils.diffusion import DiffusionScheduler


class D3PMScheduler(DiffusionScheduler):
    """Discrete diffusion scheduler with uniform corruption (D3PM, absorbing-less)."""

    class D3PMSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the D3PMScheduler class."""

        method: str = "D3PM"
        num_train_timesteps: int = 1000
        beta_start: float = 1e-4
        beta_end: float = 0.02
        beta_schedule: Literal["linear", "cosine"] = "linear"
        transition_mode: Literal["uniform", "absorbing"] = "uniform"
        num_classes: int = 2 # number of classes in the discrete data excluding absorbing state

    def __init__(self, config: D3PMSchedulerConfig) -> None:
        self.config = config
        if self.config.transition_mode == "absorbing":
            self.num_classes = self.config.num_classes + 1
            self.mask_class = self.config.num_classes
        else:
            self.num_classes = self.config.num_classes
            self.mask_class = -1
        self._setup_betas()

    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        steps = timesteps
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + 0.008) / 1.008 * (pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

    def _setup_betas(self) -> None:
        cfg = self.config
        if cfg.beta_schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
        elif cfg.beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(cfg.num_train_timesteps)
        else:
            msg = f"Unknown beta schedule {cfg.beta_schedule}"
            raise ValueError(msg)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample noise timesteps uniformly."""
        return torch.randint(
            low=0,
            high=self.config.num_train_timesteps,
            size=(batch_size,),
        )

    def alpha_bar(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Cumulative product of alphas at given timesteps."""
        idx = timesteps.long().clamp(max=self.config.num_train_timesteps - 1)
        return self.alphas_cumprod.to(timesteps.device)[idx]

    def beta_at(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Beta at given timesteps."""
        idx = timesteps.long().clamp(max=self.config.num_train_timesteps - 1)
        return self.betas.to(timesteps.device)[idx]

    def input_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the output scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the skip scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the loss weighting term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def noise_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the noise conditioning term."""
        return sigma.float() / float(self.config.num_train_timesteps)

    def sampling_time_steps(self, num_steps: int) -> torch.Tensor:
        """Generate a schedule of time steps for sampling."""
        step = max(1, self.config.num_train_timesteps // num_steps)
        times = torch.arange(
            self.config.num_train_timesteps - 1,
            -1,
            -step,
            dtype=torch.long,
        )
        if times[-1] != 0:
            times = torch.cat([times, torch.zeros(1, dtype=torch.long)])
        return times

    def sampling_schedule(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise levels for sampling."""
        return torch.clamp(time_steps, min=0.0, max=self.config.num_train_timesteps - 1)

    def sampling_schedule_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise level derivatives for sampling."""
        return torch.zeros_like(time_steps)

    def sampling_scale(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factors for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factor derivatives for sampling."""
        return torch.zeros_like(time_steps)

    def forward_prob(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate closed form q(x_t | x0) with rate alpha_bar(t)."""
        alpha_bar_t = self.alpha_bar(t).to(x0.device)
        alpha_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        alpha_bar_t = alpha_bar_t.view(alpha_shape)

        if self.config.transition_mode == "uniform":
            one_hot = torch.nn.functional.one_hot(
                x0,
                num_classes=self.num_classes,
            ).to(torch.float32)
            uniform = torch.full_like(one_hot, 1.0 / float(self.num_classes))
            return alpha_bar_t * one_hot + (1.0 - alpha_bar_t) * uniform

        if self.config.transition_mode == "absorbing":
            one_hot = torch.nn.functional.one_hot(
                x0,
                num_classes=self.num_classes,
            ).to(torch.float32)
            mask = torch.full_like(x0, self.mask_class)
            mask_hot = torch.nn.functional.one_hot(
                mask,
                num_classes=self.num_classes,
            ).to(torch.float32)
            return alpha_bar_t * one_hot + (1.0 - alpha_bar_t) * mask_hot

        msg = f"Unknown transition_mode {self.config.transition_mode}"
        raise ValueError(msg)


    def q_posterior(
        self,
        xt: torch.Tensor,
        logits_x0: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        ) -> torch.Tensor:
        """Posterior q(x_{t_prev} | x_t, x0) allowing strided timesteps."""
        alpha_bar_t = self.alpha_bar(t).to(xt.device)
        alpha_bar_prev = self.alpha_bar(torch.clamp(t_prev, min=0)).to(xt.device)
        # Effective single-step parameters when we skip >1 step (forward alpha = alpha_bar_t / alpha_bar_{t_prev})
        alpha_t = (alpha_bar_t / alpha_bar_prev).clamp(0.0, 1.0)
        beta_t = 1.0 - alpha_t
        w_t = alpha_bar_prev * beta_t / (1.0 - alpha_bar_t)

        if self.config.transition_mode == "uniform":
            pass

        elif self.config.transition_mode == "absorbing":
            mask = self.mask_class
            xt_one_hot = torch.nn.functional.one_hot(xt, num_classes=self.num_classes).float()
            probs_x0 = torch.softmax(logits_x0, dim=-1)
            probs = xt_one_hot.clone()

            mask_pos = (xt == mask).unsqueeze(-1)  # (B, L, 1)

            probs_mask = w_t * probs_x0  # (B, L, C-1)

            probs_mask = torch.concat([
                probs_mask,
                torch.ones_like(probs_mask[..., :1]) * (1.0 - w_t).unsqueeze(-1),
            ], dim=-1,
            )

            probs = torch.where(mask_pos, probs_mask, probs)
        else:
            msg = f"Unknown transition_mode {self.config.transition_mode}"
            raise ValueError(msg)

        probs = torch.clamp(probs, min=0.0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Map discrete timestep t to a continuous 'sigma' used by shared codepaths.

        We define sigma(t) = -log(alpha_bar(t)), so that exp(-sigma(t)) = alpha_bar(t).
        This makes forward_probs() consistent with forward_prob() for D3PM.
        """
        a_bar = self.alpha_bar(t).to(dtype=torch.float32)  # in (0, 1]
        return -torch.log(a_bar.clamp_min(1e-12))

    def sigma_derivative(self, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """Finite difference for sigma used by Euler-like samplers.

        For discrete timesteps this is just sigma(t) - sigma(t_next).
        """
        return self.sigma(t) - self.sigma(t_next)

    def forward_probs(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate forward probabilities p_t|0 for uniform or absorbing transitions."""
        sigma_t = self.sigma(t).to(device=x0.device, dtype=torch.float32)
        sigma_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        sigma_t = sigma_t.view(sigma_shape)

        if self.config.transition_mode == "uniform":
            decay = torch.exp(-sigma_t)
            one_hot = torch.nn.functional.one_hot(
                x0,
                num_classes=self.num_classes,
            ).to(torch.float32)
            uniform = torch.full_like(one_hot, 1.0 / float(self.num_classes))
            return decay * one_hot + (1.0 - decay) * uniform

        if self.config.transition_mode == "absorbing":
            decay = torch.exp(-sigma_t)
            one_hot = torch.nn.functional.one_hot(
                x0,
                num_classes=self.num_classes,
            ).to(torch.float32)
            mask = torch.full_like(x0, self.mask_class)
            mask_hot = torch.nn.functional.one_hot(
                mask,
                num_classes=self.num_classes,
            ).to(torch.float32)
            return decay * one_hot + (1.0 - decay) * mask_hot

        msg = f"Unknown transition_mode {self.config.transition_mode}"
        raise ValueError(msg)

    def base_q(
        self,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate generator Q entries for D3PM (rate 1)."""
        device = xt.device
        shape = (*xt.shape, self.num_classes)
        q = torch.zeros(shape, device=device, dtype=torch.float32)

        if self.config.transition_mode == "uniform":
            off_diag = 1.0 / float(max(1, self.num_classes - 1))
            q[:] = off_diag
            idx = xt.unsqueeze(-1)
            q.scatter_(-1, idx, -torch.ones_like(idx, dtype=torch.float32))
            return q

        if self.config.transition_mode == "absorbing":
            mask = self.mask_class
            # transitions to mask at rate 1, diag -1 for non-mask
            mask_tensor = torch.full_like(xt, mask)
            q.scatter_(-1, mask_tensor.unsqueeze(-1), torch.ones_like(mask_tensor, dtype=torch.float32))
            diag_updates = torch.where(
                xt == mask_tensor,
                torch.zeros_like(xt, dtype=torch.float32),
                -torch.ones_like(xt, dtype=torch.float32),
            )
            q.scatter_(-1, xt.unsqueeze(-1), diag_updates.unsqueeze(-1))
            return q

        msg = f"Unknown transition_mode {self.config.transition_mode}"
        raise ValueError(msg)


class SEDDScheduler(DiffusionScheduler):
    """Scheduler for SEDD (score entropy discrete diffusion)."""

    class SEDDSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the SEDDScheduler class."""

        method: str = "SEDD"
        num_train_timesteps: int = 1000
        sigma_min: float = 1e-4
        sigma_max: float = 20.0
        schedule: Literal["geometric", "linear"] = "geometric"
        transition_mode: Literal["uniform", "absorbing"] = "uniform"
        num_classes: int = 2  # number of classes in the discrete data excluding absorbing state

    def __init__(self, config: SEDDSchedulerConfig) -> None:
        self.config = config
        if self.config.transition_mode == "absorbing":
            self.num_classes = self.config.num_classes + 1
            self.mask_class = self.config.num_classes
        else:
            self.mask_class = -1

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample noise magnitudes uniformly between sigma_min and sigma_max."""
        return torch.rand(batch_size)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise strength increasing from 0 to sigma_max."""
        if self.config.schedule == "geometric":
            return (
                self.config.sigma_min ** (1.0 - t)
                * self.config.sigma_max ** t
            )
        if self.config.schedule == "linear":
            return self.config.sigma_min + t * (
                self.config.sigma_max - self.config.sigma_min
            )
        msg = f"Unknown SEDD schedule {self.config.schedule}"
        raise ValueError(msg)

    def sigma_derivative(self, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """Finite difference Δσ used for Euler stepping."""
        return self.sigma(t) - self.sigma(t_next)

    def input_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the output scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the skip scaling term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the loss weighting term."""
        return torch.ones_like(sigma, dtype=torch.float32)

    def noise_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the noise conditioning term."""
        return torch.log1p(sigma)

    def sampling_time_steps(self, num_steps: int) -> torch.Tensor:
        """Generate a schedule of time steps for sampling."""
        return torch.linspace(1.0, 0.0, steps=num_steps + 1)

    def sampling_schedule(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise levels for sampling."""
        return time_steps

    def sampling_schedule_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise level derivatives for sampling."""
        return torch.zeros_like(time_steps)

    def sampling_scale(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factors for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factor derivatives for sampling."""
        return torch.zeros_like(time_steps)

    def forward_prob(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate closed form q(x_t | x0) with rate exp(-sigma(t))."""
        sigma_t = self.sigma(t).to(x0.device)
        sigma_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        sigma_t = sigma_t.view(sigma_shape)
        decay = torch.exp(-sigma_t)
        one_hot = torch.nn.functional.one_hot(
            x0,
            num_classes=self.num_classes,
        ).to(torch.float32)

        if self.config.transition_mode == "uniform":
            uniform = torch.full_like(one_hot, 1.0 / float(self.num_classes))
            return decay * one_hot + (1.0 - decay) * uniform

        if self.config.transition_mode == "absorbing":
            mask = torch.full_like(x0, self.mask_class)
            mask_hot = torch.nn.functional.one_hot(
                mask,
                num_classes=self.num_classes,
            ).to(torch.float32)
            return decay * one_hot + (1.0 - decay) * mask_hot

        msg = f"Unknown transition_mode {self.config.transition_mode}"
        raise ValueError(msg)

    def base_q(
        self,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate generator Q entries for SEDD (rate 1)."""
        device = xt.device
        shape = (*xt.shape, self.num_classes)
        q = torch.zeros(shape, device=device, dtype=torch.float32)

        if self.config.transition_mode == "uniform":
            off_diag = 1.0 / float(max(1, self.num_classes - 1))
            q[:] = off_diag
            q.scatter_(-1, xt.unsqueeze(-1), -torch.ones_like(xt, dtype=torch.float32).unsqueeze(-1))
            return q

        if self.config.transition_mode == "absorbing":
            mask = self.mask_class
            mask_tensor = torch.full_like(xt, mask)
            q.scatter_(-1, mask_tensor.unsqueeze(-1), torch.ones_like(mask_tensor, dtype=torch.float32))
            diag_updates = torch.where(
                xt == mask_tensor,
                torch.zeros_like(xt, dtype=torch.float32),
                -torch.ones_like(xt, dtype=torch.float32),
            )
            q.scatter_(-1, xt.unsqueeze(-1), diag_updates.unsqueeze(-1))
            return q

        msg = f"Unknown transition_mode {self.config.transition_mode}"
        raise ValueError(msg)

