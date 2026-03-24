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
    def sample_noise(self, batch_size: int) -> Float[torch.Tensor, ...]:
        """Noise magnitude at time t."""

    @abstractmethod
    def input_scale(
        self,
        sigma: Float[torch.Tensor, ...],
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

    def sample_noise(self, batch_size: int) -> Float[torch.Tensor, ...]:
        """Sample noise from a exp Normal distribution."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
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
    ) -> Float[torch.Tensor, ...]:
        """Compute the input scaling term."""
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
