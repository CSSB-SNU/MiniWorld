from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from pydantic import BaseModel


# ruff: noqa: E501
class DiffusionScheduler(ABC):
    """Maps a time index t -> noise level(s), scaling factors, loss weights, etc."""

    class DiffusionSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

    @abstractmethod
    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> (
        Float[torch.Tensor, ...]
        | tuple[
            Float[torch.Tensor, ...],
            Float[torch.Tensor, ...],
            Float[torch.Tensor, ...],
        ]
    ):
        """Noise magnitude at time t."""

    @abstractmethod
    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """How to scale the clean input into the noisy domain."""

    @abstractmethod
    def output_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """How to scale the model's raw prediction back to the data domain."""

    @abstractmethod
    def skip_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """How to scale the noixy x to the data domain."""

    @abstractmethod
    def loss_weight(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Weight to apply to loss at noise level sigma."""

    @abstractmethod
    def noise_condition(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Conditioning for the noise prediction."""

    @abstractmethod
    def sampling_time_steps(self, num_steps: int) -> Float[torch.Tensor, ...]:
        """Generate a schedule of time steps for sampling."""

    @abstractmethod
    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)

    @abstractmethod
    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt

    @abstractmethod
    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)

    @abstractmethod
    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt


class EDMScheduler(DiffusionScheduler):
    """EDM scheduler implementing DiffusionScheduler.

    Revised based on AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma) ~ N(P_mean, P_std).
      - Loss weight: (sigma^2+sigma_data^2)/(sigma*sigma_data)^2.
    """

    class EDMSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the EDMScheduler class."""

        method: str = "AF3"
        P_mean: float = -1.2
        P_std: float = 1.5
        sigma_data: float = 16.0
        sigma_max: float = 160.0
        sigma_min: float = 4e-4
        rho: float = 7.0
        use_time_augmentation: bool = True

    def __init__(self, config: EDMSchedulerConfig) -> None:
        self.config = config

    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> Float[torch.Tensor, ...]:
        """Sample noise from a exp Normal distribution."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
        if uniform:
            num_uniform_steps = max(batch_size - 1, 1)
            return self.sampling_time_steps(num_uniform_steps)[:batch_size]
        if self.config.use_time_augmentation:
            u = torch.rand(batch_size)
        else:
            u = torch.rand(1).expand(batch_size)
        normal = torch.distributions.Normal(0.0, 1.0)

        sigma = self.config.sigma_data * torch.exp(
            self.config.P_mean + self.config.P_std * normal.icdf(u),
        )
        return torch.clamp(
            sigma,
            min=self.config.sigma_min * self.config.sigma_data,
            max=self.config.sigma_max * self.config.sigma_data,
        )

    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Compute the input scaling term."""
        del sigma_translation
        return 1.0 / torch.sqrt(sigma**2 + self.config.sigma_data**2)

    def output_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the output scaling term."""
        sd = self.config.sigma_data
        return (sigma * sd) / (sigma**2 + sd**2)

    def skip_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the skip scaling term."""
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the loss weighting term."""
        sd = self.config.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def noise_condition(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the noise conditioning term."""
        return sigma.log() / 4.0

    def sampling_time_steps(self, num_steps: int) -> Float[torch.Tensor, ...]:
        """Generate a schedule of time steps for sampling."""
        time_steps = torch.empty(num_steps + 1)
        t = torch.linspace(0.0, 1.0, steps=num_steps)
        sigma_max_r = self.config.sigma_max ** (1.0 / self.config.rho)
        sigma_min_r = self.config.sigma_min ** (1.0 / self.config.rho)
        time_steps[:-1] = (
            self.config.sigma_data
            * (sigma_max_r + t * (sigma_min_r - sigma_max_r)) ** self.config.rho
        )
        time_steps[-1] = 0.0
        return time_steps

    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)
        return time_steps

    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt
        return torch.ones_like(time_steps)

    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt
        return torch.zeros_like(time_steps)


class DecoupledEDMScheduler(DiffusionScheduler):
    """Decoupled EDM scheduler implementing DiffusionScheduler.

    Revised based on the AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma_y) ~ N(P_mean, P_std).
      - Rotation and translation noise use their own schedules derived from sigma_y.
    """

    class DecoupledEDMSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        """Configuration for the DecoupledEDMScheduler class."""

        method: str = "AF3"
        P_mean: float = -1.2
        P_std: float = 1.5
        sigma_data: float = 16.0

        sigma_y_max: float = 160.0
        sigma_y_min: float = 4e-4
        rho_y: float = 9.0

        sigma_R_max: float = 3000  # noqa: N815
        sigma_R_min: float = 4e-4  # noqa: N815
        rho_R: float = 0.1  # noqa: N815

        sigma_T_max: float = 15.0  # noqa: N815
        sigma_T_min: float = 4e-4  # noqa: N815
        rho_T: float = 1.0  # noqa: N815

        use_time_augmentation: bool = True

    def __init__(self, config: DecoupledEDMSchedulerConfig) -> None:
        self.config = config

    def convert_to_sigma_rt(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> tuple[Float[torch.Tensor, ...], Float[torch.Tensor, ...]]:
        """Convert the coordinate noise level into rotation and translation noise."""
        eps_mask = sigma <= 1e-8

        def power(
            sigma_min: float,
            sigma_max: float,
            rho: float,
        ) -> tuple[float, float]:
            return sigma_min ** (1.0 / rho), sigma_max ** (1.0 / rho)

        b_y, a_y = power(
            self.config.sigma_y_min * self.config.sigma_data,
            self.config.sigma_y_max * self.config.sigma_data,
            self.config.rho_y,
        )
        b_rotation, a_rotation = power(
            self.config.sigma_R_min,
            self.config.sigma_R_max,
            self.config.rho_R,
        )
        b_translation, a_translation = power(
            self.config.sigma_T_min * self.config.sigma_data,
            self.config.sigma_T_max * self.config.sigma_data,
            self.config.rho_T,
        )

        kappa_rotation = (b_rotation - a_rotation) / (b_y - a_y)
        kappa_translation = (b_translation - a_translation) / (b_y - a_y)
        c_rotation = -a_y * (b_rotation - a_rotation) / (b_y - a_y) + a_rotation
        c_translation = (
            -a_y * (b_translation - a_translation) / (b_y - a_y) + a_translation
        )
        sigma_base = sigma ** (1.0 / self.config.rho_y)
        sigma_rotation = (kappa_rotation * sigma_base + c_rotation) ** self.config.rho_R
        sigma_translation = (
            kappa_translation * sigma_base + c_translation
        ) ** self.config.rho_T

        sigma_rotation = torch.where(
            eps_mask,
            torch.full_like(sigma_rotation, 1e-8),
            sigma_rotation,
        )
        sigma_translation = torch.where(
            eps_mask,
            torch.full_like(sigma_translation, 1e-8),
            sigma_translation,
        )
        return sigma_rotation, sigma_translation

    def sample_noise(
        self,
        batch_size: int,
        uniform: bool = False,
    ) -> tuple[
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
    ]:
        """Sample decoupled noise magnitudes for coordinates, rotation, and translation."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
        if self.config.use_time_augmentation:
            u = torch.rand(batch_size)
        else:
            u = torch.rand(1).expand(batch_size)
        if uniform:
            num_uniform_steps = max(batch_size - 1, 1)
            sigma = self.sampling_time_steps(num_uniform_steps)[:batch_size]
        else:
            normal = torch.distributions.Normal(0.0, 1.0)
            sigma = self.config.sigma_data * torch.exp(
                self.config.P_mean + self.config.P_std * normal.icdf(u),
            )

        sigma_y = torch.clamp(
            sigma,
            min=self.config.sigma_y_min * self.config.sigma_data,
            max=self.config.sigma_y_max * self.config.sigma_data,
        )
        sigma_rotation, sigma_translation = self.convert_to_sigma_rt(sigma_y)
        return sigma_y, sigma_rotation, sigma_translation

    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
        sigma_translation: Float[torch.Tensor, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Compute the input scaling term."""
        if sigma_translation is None:
            msg = "sigma_translation must be provided for decoupled EDM input scaling."
            raise ValueError(msg)
        return 1.0 / torch.sqrt(
            sigma**2 + sigma_translation**2 + self.config.sigma_data**2,
        )

    def output_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the output scaling term."""
        sd = self.config.sigma_data
        return (sigma * sd) / (sigma**2 + sd**2)

    def skip_scale(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the skip scaling term."""
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the loss weighting term."""
        sd = self.config.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def noise_condition(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Compute the noise conditioning term."""
        return sigma.log() / 4.0

    def sampling_time_steps(self, num_steps: int) -> Float[torch.Tensor, ...]:
        """Generate a schedule of time steps for sampling."""
        time_steps = torch.empty(num_steps + 1)
        t = torch.linspace(0.0, 1.0, steps=num_steps)
        sigma_max_r = self.config.sigma_y_max ** (1.0 / self.config.rho_y)
        sigma_min_r = self.config.sigma_y_min ** (1.0 / self.config.rho_y)
        time_steps[:-1] = (
            self.config.sigma_data
            * (sigma_max_r + t * (sigma_min_r - sigma_max_r)) ** self.config.rho_y
        )
        time_steps[-1] = 0.0
        return time_steps

    def sampling_schedule(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise levels for sampling."""
        return time_steps

    def sampling_schedule_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of noise level derivatives for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factors for sampling."""
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(
        self,
        time_steps: Float[torch.Tensor, ...],
    ) -> Float[torch.Tensor, ...]:
        """Generate a schedule of scaling factor derivatives for sampling."""
        return torch.zeros_like(time_steps)
