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
        rho_y: float = 7.0
        # Phase boundaries expressed as progress through the full sigma_y range
        # in EDM-transformed space. Larger values push the boundary closer to the
        # low-noise regime.
        sigma_y_phase_1_progress: float = 0.5
        sigma_y_phase_2_progress: float = 0.7
        phase_1_fraction: float = 0.35
        phase_2_fraction: float = 0.35

        sigma_R_max: float = 32.0  # noqa: N815
        sigma_R_min: float = 4e-4  # noqa: N815
        rho_R: float = 1.5  # noqa: N815

        sigma_T_max: float = 8.0  # noqa: N815
        sigma_T_min: float = 4e-4  # noqa: N815
        rho_T: float = 1.0  # noqa: N815

        use_time_augmentation: bool = True

    def __init__(self, config: DecoupledEDMSchedulerConfig) -> None:
        self.config = config
        if not (0.0 < self.config.sigma_y_phase_1_progress < 1.0):
            msg = "sigma_y_phase_1_progress must lie in (0, 1)."
            raise ValueError(msg)
        if not (0.0 < self.config.sigma_y_phase_2_progress < 1.0):
            msg = "sigma_y_phase_2_progress must lie in (0, 1)."
            raise ValueError(msg)
        if self.config.sigma_y_phase_1_progress >= self.config.sigma_y_phase_2_progress:
            msg = (
                "sigma_y_phase_1_progress must be smaller than sigma_y_phase_2_progress."
            )
            raise ValueError(msg)
        if not (0.0 < self.config.phase_1_fraction < 1.0):
            msg = "phase_1_fraction must lie in (0, 1)."
            raise ValueError(msg)
        if not (0.0 < self.config.phase_2_fraction < 1.0):
            msg = "phase_2_fraction must lie in (0, 1)."
            raise ValueError(msg)
        if self.config.phase_1_fraction + self.config.phase_2_fraction >= 1.0:
            msg = "phase_1_fraction + phase_2_fraction must be less than 1."
            raise ValueError(msg)

    @staticmethod
    def _interpolate_sigma(
        progress: Float[torch.Tensor, ...],
        sigma_start: float,
        sigma_end: float,
        rho: float,
    ) -> Float[torch.Tensor, ...]:
        """Interpolate between sigma endpoints in EDM-style transformed space."""
        sigma_start_r = sigma_start ** (1.0 / rho)
        sigma_end_r = sigma_end ** (1.0 / rho)
        return (sigma_start_r + progress * (sigma_end_r - sigma_start_r)) ** rho

    @staticmethod
    def _interpolation_progress(
        sigma: Float[torch.Tensor, ...],
        sigma_start: float,
        sigma_end: float,
        rho: float,
    ) -> Float[torch.Tensor, ...]:
        """Invert `_interpolate_sigma` for monotone decreasing schedules."""
        sigma_start_r = sigma_start ** (1.0 / rho)
        sigma_end_r = sigma_end ** (1.0 / rho)
        sigma_r = torch.clamp(sigma, min=0.0) ** (1.0 / rho)
        return torch.clamp(
            (sigma_r - sigma_start_r) / (sigma_end_r - sigma_start_r),
            min=0.0,
            max=1.0,
        )

    def _sigma_y_phase_boundaries(self) -> tuple[float, float, float, float]:
        """Return the full sigma_y range and the two phase boundaries."""
        sigma_y_max = self.config.sigma_y_max * self.config.sigma_data
        sigma_y_min = self.config.sigma_y_min * self.config.sigma_data
        sigma_y_phase_1_end = float(
            self._interpolate_sigma(
                torch.tensor(self.config.sigma_y_phase_1_progress),
                sigma_y_max,
                sigma_y_min,
                self.config.rho_y,
            ),
        )
        sigma_y_phase_2_end = float(
            self._interpolate_sigma(
                torch.tensor(self.config.sigma_y_phase_2_progress),
                sigma_y_max,
                sigma_y_min,
                self.config.rho_y,
            ),
        )
        return sigma_y_max, sigma_y_min, sigma_y_phase_1_end, sigma_y_phase_2_end

    def convert_to_sigma_rt(
        self,
        sigma: Float[torch.Tensor, ...],
    ) -> tuple[Float[torch.Tensor, ...], Float[torch.Tensor, ...]]:
        """Convert sigma_y into a three-phase rigid-body schedule.

        Phase 1:
          - sigma_y decreases from sigma_y_max to the first progress boundary
          - sigma_rotation / sigma_translation stay at their maxima
        Phase 2:
          - sigma_y traverses the middle progress band
          - sigma_rotation / sigma_translation decay to their minima
        Phase 3:
          - sigma_rotation / sigma_translation stay near zero
          - sigma_y decreases to sigma_y_min for high-resolution refinement
        """
        eps_mask = sigma <= 1e-8
        sigma_y_max, sigma_y_min, sigma_y_phase_1_end, sigma_y_phase_2_end = (
            self._sigma_y_phase_boundaries()
        )
        sigma_y = torch.clamp(sigma, min=sigma_y_min, max=sigma_y_max)

        sigma_translation_max = self.config.sigma_T_max * self.config.sigma_data
        sigma_translation_min = self.config.sigma_T_min * self.config.sigma_data

        sigma_rotation = torch.full_like(sigma_y, self.config.sigma_R_min)
        sigma_translation = torch.full_like(sigma_y, sigma_translation_min)

        phase_1_mask = sigma_y >= sigma_y_phase_1_end
        phase_2_mask = (sigma_y < sigma_y_phase_1_end) & (sigma_y >= sigma_y_phase_2_end)

        sigma_rotation = torch.where(
            phase_1_mask,
            torch.full_like(sigma_rotation, self.config.sigma_R_max),
            sigma_rotation,
        )
        sigma_translation = torch.where(
            phase_1_mask,
            torch.full_like(sigma_translation, sigma_translation_max),
            sigma_translation,
        )

        if phase_2_mask.any():
            phase_2_progress = self._interpolation_progress(
                sigma_y[phase_2_mask],
                sigma_y_phase_1_end,
                sigma_y_phase_2_end,
                self.config.rho_y,
            )
            sigma_rotation_phase_2 = self._interpolate_sigma(
                phase_2_progress,
                self.config.sigma_R_max,
                self.config.sigma_R_min,
                self.config.rho_R,
            )
            sigma_translation_phase_2 = self._interpolate_sigma(
                phase_2_progress,
                sigma_translation_max,
                sigma_translation_min,
                self.config.rho_T,
            )
            sigma_rotation = sigma_rotation.clone()
            sigma_translation = sigma_translation.clone()
            sigma_rotation[phase_2_mask] = sigma_rotation_phase_2
            sigma_translation[phase_2_mask] = sigma_translation_phase_2

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
        """Generate a three-phase sigma_y schedule for sampling/training."""
        time_steps = torch.empty(num_steps + 1)
        sigma_y_max, sigma_y_min, sigma_y_phase_1_end, sigma_y_phase_2_end = (
            self._sigma_y_phase_boundaries()
        )

        progress = torch.linspace(0.0, 1.0, steps=num_steps)
        phase_1_end = self.config.phase_1_fraction
        phase_2_end = self.config.phase_1_fraction + self.config.phase_2_fraction

        sigma_y = torch.empty(num_steps)
        phase_1_mask = progress <= phase_1_end
        phase_2_mask = (progress > phase_1_end) & (progress <= phase_2_end)
        phase_3_mask = progress > phase_2_end

        phase_1_progress = progress[phase_1_mask] / phase_1_end
        sigma_y[phase_1_mask] = self._interpolate_sigma(
            phase_1_progress,
            sigma_y_max,
            sigma_y_phase_1_end,
            self.config.rho_y,
        )

        if phase_2_mask.any():
            phase_2_progress = (
                progress[phase_2_mask] - phase_1_end
            ) / self.config.phase_2_fraction
            sigma_y[phase_2_mask] = self._interpolate_sigma(
                phase_2_progress,
                sigma_y_phase_1_end,
                sigma_y_phase_2_end,
                self.config.rho_y,
            )

        if phase_3_mask.any():
            phase_3_progress = (progress[phase_3_mask] - phase_2_end) / (
                1.0 - phase_2_end
            )
            sigma_y[phase_3_mask] = self._interpolate_sigma(
                phase_3_progress,
                sigma_y_phase_2_end,
                sigma_y_min,
                self.config.rho_y,
            )

        time_steps[:-1] = sigma_y
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
