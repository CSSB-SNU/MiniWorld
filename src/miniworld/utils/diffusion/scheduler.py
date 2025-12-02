from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import LogFormatter, LogLocator
from pydantic import BaseModel


class DiffusionScheduler(ABC):
    """Maps a time index t -> noise level(s), scaling factors, loss weights, etc."""

    class DiffusionSchedulerConfig(BaseModel):
        """Configuration for the DiffusionScheduler class."""

        method: str = "EDM"

    @abstractmethod
    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Noise magnitude at time t."""

    @abstractmethod
    def input_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """How to scale the clean input into the noisy domain."""

    @abstractmethod
    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """How to scale the model's raw prediction back to the data domain."""

    @abstractmethod
    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """How to scale the noixy x to the data domain."""

    @abstractmethod
    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Weight to apply to loss at noise level sigma."""

    @abstractmethod
    def noise_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Conditioning for the noise prediction."""

    @abstractmethod
    def sampling_time_steps(self, num_steps: int) -> torch.Tensor:
        """Generate a schedule of time steps for sampling."""

    @abstractmethod
    def sampling_schedule(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)

    @abstractmethod
    def sampling_schedule_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt

    @abstractmethod
    def sampling_scale(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)

    @abstractmethod
    def sampling_scale_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
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

    def sample_noise(self, batch_size: int) -> torch.Tensor:
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
        return torch.clamp(sigma, min=self.config.sigma_min * self.config.sigma_data, max=self.config.sigma_max*self.config.sigma_data)

    def input_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling term."""
        return 1.0 / torch.sqrt(sigma**2 + self.config.sigma_data**2)

    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the output scaling term."""
        sd = self.config.sigma_data
        return (sigma * sd) / (sigma**2 + sd**2)

    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the skip scaling term."""
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the loss weighting term."""
        sd = self.config.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def noise_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the noise conditioning term."""
        return sigma.log() / 4.0

    def sampling_time_steps(self, num_steps: int) -> torch.Tensor:
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

    def sampling_schedule(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)
        return time_steps

    def sampling_schedule_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt
        return torch.ones_like(time_steps)

    def sampling_scale(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt
        return torch.zeros_like(time_steps)

class DecoupledEDMScheduler(DiffusionScheduler):
    """EDM scheduler implementing DiffusionScheduler.

    Revised based on AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma) ~ N(P_mean, P_std).
      - Loss weight: (sigma^2+sigma_data^2)/(sigma*sigma_data)^2.
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

        sigma_rotation_max: float = 4
        sigma_rotation_min: float = 4e-4
        rho_rotation: float = 1.5

        sigma_translation_max: float = 8
        sigma_translation_min: float = 4e-4
        rho_translation: float = 1.0


        use_time_augmentation: bool = True

    def __init__(self, config: DecoupledEDMSchedulerConfig) -> None:
        self.config = config

    def convert_to_sigmaRT(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: N802
        """Convert sigma to (R, T) space."""
        eps_mask = sigma <= 1e-8

        def power(sigma_min: float, sigma_max: float, rho: float) -> tuple[float, float]:
            return sigma_min ** (1/rho), sigma_max ** (1/rho)

        b_y, a_y = power(self.config.sigma_y_min * self.config.sigma_data, self.config.sigma_y_max* self.config.sigma_data, self.config.rho_y)
        b_R, a_R = power(self.config.sigma_rotation_min, self.config.sigma_rotation_max, self.config.rho_rotation)
        b_T, a_T = power(self.config.sigma_translation_min* self.config.sigma_data, self.config.sigma_translation_max* self.config.sigma_data, self.config.rho_translation)

        kappa_R = (b_R - a_R) / (b_y - a_y)
        kappa_T = (b_T - a_T) / (b_y - a_y)
        c_R = - a_y * (b_R - a_R) / (b_y - a_y) + a_R
        c_T = - a_y * (b_T - a_T) / (b_y - a_y) + a_T
        sigma_rotation = (kappa_R * sigma ** (1/self.config.rho_y) + c_R) ** self.config.rho_rotation
        sigma_translation = (kappa_T * sigma ** (1/self.config.rho_y) + c_T) ** self.config.rho_translation

        sigma_rotation[eps_mask] = 1e-8
        sigma_translation[eps_mask] = 1e-8

        return sigma_rotation, sigma_translation

    def sample_noise(self, batch_size: int, uniform : bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample noise from a exp Normal distribution."""
        if batch_size <= 0:
            msg = f"Batch size should be greater than 0, got {batch_size}"
            raise ValueError(msg)
        if self.config.use_time_augmentation:
            u = torch.rand(batch_size)
        else:
            u = torch.rand(1).expand(batch_size)
        if uniform:
            sigma = self.sampling_time_steps(batch_size-1)
        else:
            normal = torch.distributions.Normal(0.0, 1.0)

            sigma = self.config.sigma_data * torch.exp(
                self.config.P_mean + self.config.P_std * normal.icdf(u),
            )
        sigma_y = torch.clamp(sigma, min=self.config.sigma_y_min * self.config.sigma_data, max=self.config.sigma_y_max*self.config.sigma_data)
        sigma_rotation, sigma_translation = self.convert_to_sigmaRT(sigma_y)
        return sigma_y, sigma_rotation, sigma_translation

    def input_scale(self, sigma_y: torch.Tensor, sigma_translation: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling term."""
        return 1.0 / torch.sqrt(sigma_y**2 + sigma_translation**2 + self.config.sigma_data**2)

    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the output scaling term."""
        sd = self.config.sigma_data
        return (sigma * sd) / (sigma**2 + sd**2)

    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the skip scaling term."""
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the loss weighting term."""
        sd = self.config.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def noise_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the noise conditioning term."""
        return sigma.log() / 4.0

    def sampling_time_steps(self, num_steps: int) -> torch.Tensor:
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

    def sampling_schedule(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise levels for sampling."""
        # t -> sigma(t)
        return time_steps

    def sampling_schedule_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of noise level derivatives for sampling."""
        #  t -> dsigma(t)/dt
        return torch.ones_like(time_steps)

    def sampling_scale(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factors for sampling."""
        # t -> scale(t)
        return torch.ones_like(time_steps)

    def sampling_scale_derivative(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generate a schedule of scaling factor derivatives for sampling."""
        # t -> dscale(t)/dt
        return torch.zeros_like(time_steps)

    def draw_sigma(self, save_path: Path, mode : Literal["inference", "train","y_to_RT"] = "inference") -> str:  # noqa: PLR0915
        """Plot sigma schedules vs normalized time t in [0,1] and save the figure.

        - y-axis is log-scaled to show the full dynamic range.
        - Curves shown: sigma_y(t), sigma_rotation(t), sigma_translation(t).
        - Uses self.sampling_time_steps() and self.convert_to_sigmaRT().

        Returns:
            The save_path for convenience.

        """
        # --- build schedule ---
        if mode == "inference":
            num_steps = 1000
            t = torch.linspace(0.0, 1.0, steps=num_steps + 1)  # includes t=1 endpoint
            sigma_y = self.sampling_time_steps(num_steps)      # length num_steps+1, ends with 0
            # Avoid log(0) at the last point for display
            eps = 1e-12
            sigma_y_plot = torch.clamp(sigma_y, min=eps)

            # map to (R,T) space
            sigma_rotation, sigma_translation = self.convert_to_sigmaRT(sigma_y_plot)

            # --- to numpy for matplotlib ---
            t_np = t.detach().cpu().numpy()
            sigy_np = sigma_y_plot.detach().cpu().numpy()
            sigR_np = torch.clamp(sigma_rotation, min=eps).detach().cpu().numpy()
            sigT_np = torch.clamp(sigma_translation, min=eps).detach().cpu().numpy()

            t_np = t_np[:-1]
            sigy_np = sigy_np[:-1]
            sigR_np = sigR_np[:-1]
            sigT_np = sigT_np[:-1]

            # --- plot ---
            fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
            ax.plot(t_np, sigy_np, label=r"$\sigma_y(t)$")
            ax.plot(t_np, sigR_np, label=r"$\sigma_rotation(t)$")
            ax.plot(t_np, sigT_np, label=r"$\sigma_translation(t)$")
            ax.set_yscale("log")
            ax.set_xlabel("t (normalized)")
            ax.set_ylabel("sigma (log scale)")
            ax.set_title(
                f"Decoupled EDM Schedule\n"
                f"sigma_data={self.config.sigma_data}, "
                f"rho_y={self.config.rho_y}, rho_rotation={self.config.rho_rotation}, rho_translation={self.config.rho_translation}",
            )
            ax.grid(visible=True, which="both", ls="--", alpha=0.3)
            ax.legend()

            # ensure directory exists and save

            save_path.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif mode == "train":
            num_samples = 10_000
            sigma_y, sigma_rotation, sigma_translation = self.sample_noise(num_samples, uniform=True)

            eps = 1e-12  # strictly positive for log-x
            y_np = torch.clamp(sigma_y, min=eps).detach().cpu().numpy()
            R_np = torch.clamp(sigma_rotation, min=eps).detach().cpu().numpy()
            T_np = torch.clamp(sigma_translation, min=eps).detach().cpu().numpy()

            sd = self.config.sigma_data
            # Per-subplot limits from your config (what you asked for)
            limits = [
                (self.config.sigma_y_min * sd, self.config.sigma_y_max * sd, y_np, r"$\sigma_y$"),
                (self.config.sigma_rotation_min,      self.config.sigma_rotation_max,      R_np, r"$\sigma_rotation$"),
                (self.config.sigma_translation_min * sd, self.config.sigma_translation_max * sd, T_np, r"$\sigma_translation$"),
            ]

            # No shared x-axis → each subplot uses its own log-x range
            fig, axs = plt.subplots(3, 1, figsize=(7.5, 10.0), dpi=140)

            for ax, (_xmin, _xmax, arr, title) in zip(axs, limits, strict=False):
                xmin = max(float(_xmin), eps)
                xmax = max(float(_xmax), xmin * (1.0 + 1e-9))  # avoid equal min/max
                bins = np.logspace(np.log10(xmin), np.log10(xmax), 40)

                ax.hist(arr, bins=bins, density=False)
                ax.set_title(title)
                ax.set_xscale("log")
                ax.set_xlim(xmin, xmax)

                # Nice log-x ticks
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_major_formatter(LogFormatter(base=10))
                ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))

            axs[-1].set_xlabel("Value (log-spaced bins)")
            plt.tight_layout()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif mode == "y_to_RT":
            num_steps = 1000
            # sigma_y schedule in [sigma_y_max ... -> 0], last element is 0
            sigma_y_full = self.sampling_time_steps(num_steps)
            eps = 1e-12
            # drop the trailing 0 and clamp for log
            sigma_y = torch.clamp(sigma_y_full[:-1], min=eps)

            # map to (R,T)
            sigma_rotation, sigma_translation = self.convert_to_sigmaRT(sigma_y)

            # to numpy
            x_y = sigma_y.detach().cpu().numpy()
            y_R = torch.clamp(sigma_rotation, min=eps).detach().cpu().numpy()
            y_T = torch.clamp(sigma_translation, min=eps).detach().cpu().numpy()

            # figure
            fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
            ax.plot(x_y, y_R, label=r"$\sigma_rotation(\sigma_y)$")
            ax.plot(x_y, y_T, label=r"$\sigma_translation(\sigma_y)$")

            # log-log for both axes
            ax.set_xscale("log")
            ax.set_yscale("log")

            # nice limits from config (avoid 0)
            sd = self.config.sigma_data
            xmin = max(self.config.sigma_y_min * sd, eps)
            xmax = max(self.config.sigma_y_max * sd, xmin * (1 + 1e-9))
            ax.set_xlim(xmin, xmax)

            # y-limits from config (cover both R and T)
            ymin = min(self.config.sigma_rotation_min, self.config.sigma_translation_min * sd, y_R.min(), y_T.min())
            ymax = max(self.config.sigma_rotation_max, self.config.sigma_translation_max * sd, y_R.max(), y_T.max())
            ymin = max(ymin, eps)
            ymax = max(ymax, ymin * (1 + 1e-9))
            ax.set_ylim(ymin, ymax)

            ax.set_xlabel(r"$\sigma_y$")
            ax.set_ylabel(r"$\sigma_{R/T}$")
            ax.set_title(
                "Mapping from $\\sigma_y$ to $(\\sigma_rotation, \\sigma_translation)$\n"
                f"sigma_data={sd}, rho_y={self.config.rho_y}, "
                f"rho_rotation={self.config.rho_rotation}, rho_translation={self.config.rho_translation}",
            )
            ax.grid(visible=True, which="both", ls="--", alpha=0.3)
            ax.legend()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        return str(save_path)
