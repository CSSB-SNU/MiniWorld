import torch
from team_gm.utils.scheduler import DiffusionScheduler


class DecoupledEDMScheduler(DiffusionScheduler):
    """
    EDM scheduler implementing DiffusionScheduler.
    Revised based on AF3 paper:
      - sigma_data set to 16.
      - Noise distribution: ln((1/sigma_data)*sigma) ~ N(P_mean, P_std).
      - Loss weight: (sigma^2+sigma_data^2)/(sigma*sigma_data)^2.
    """

    class DecoupledEDMSchedulerConfig(DiffusionScheduler.DiffusionSchedulerConfig):
        method: str = "AF3"
        P_mean: float = -1.2
        P_std: float = 1.5
        sigma_data: float = 16.0

        sigma_y_max: float = 160.0
        sigma_y_min: float = 4e-4
        rho_y: float = 9.0

        sigma_R_max: float = 4.0
        sigma_R_min: float = 4e-4
        rho_R: float = 1.5

        sigma_T_max: float = 8.0
        sigma_T_min: float = 4e-4
        rho_T: float = 1.0


        use_time_augmentation: bool = True

    def __init__(self, config: DecoupledEDMSchedulerConfig):
        self.config = config

    def convert_to_sigmaRT(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert sigma to (R, T) space."""
        if sigma.norm() < 1e-12:
            return torch.zeros_like(sigma), torch.zeros_like(sigma)

        def power(sigma_min, sigma_max,rho) :
            return sigma_min ** (1/rho), sigma_max ** (1/rho)

        b_y, a_y = power(self.config.sigma_y_min * self.config.sigma_data, self.config.sigma_y_max* self.config.sigma_data, self.config.rho_y)
        b_R, a_R = power(self.config.sigma_R_min, self.config.sigma_R_max, self.config.rho_R)
        b_T, a_T = power(self.config.sigma_T_min* self.config.sigma_data, self.config.sigma_T_max* self.config.sigma_data, self.config.rho_T)

        kappa_R = (b_R - a_R) / (b_y - a_y)
        kappa_T = (b_T - a_T) / (b_y - a_y)
        c_R = - a_y * (b_R - a_R) / (b_y - a_y) + a_R
        c_T = - a_y * (b_T - a_T) / (b_y - a_y) + a_T
        sigma_R = (kappa_R * sigma ** (1/self.config.rho_y) + c_R) ** self.config.rho_R
        sigma_T = (kappa_T * sigma ** (1/self.config.rho_y) + c_T) ** self.config.rho_T

        return sigma_R, sigma_T

    def sample_noise(self, B: int, uniform : bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample noise from a exp Normal distribution."""
        assert B > 0, f"Batch size should be greater than 0, got {B}"
        if self.config.use_time_augmentation:
            u = torch.rand(B)
        else:
            u = torch.rand(1).expand(B)
        if uniform:
            sigma = self.sampling_time_steps(B-1)
        else:
            normal = torch.distributions.Normal(0.0, 1.0)

            sigma = self.config.sigma_data * torch.exp(
                self.config.P_mean + self.config.P_std * normal.icdf(u)
            )
        sigma_y = torch.clamp(sigma, min=self.config.sigma_y_min * self.config.sigma_data, max=self.config.sigma_y_max*self.config.sigma_data)
        sigma_R, sigma_T = self.convert_to_sigmaRT(sigma_y)
        return sigma_y, sigma_R, sigma_T

    def input_scale(self, sigma_y: torch.Tensor, sigma_T: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling term."""
        return 1.0 / torch.sqrt(sigma_y**2 + sigma_T**2 + self.config.sigma_data**2)

    def output_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        sd = self.config.sigma_data
        return (sigma * sd) / (sigma**2 + sd**2)

    def skip_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        sd = self.config.sigma_data
        return sd**2 / (sigma**2 + sd**2)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
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
