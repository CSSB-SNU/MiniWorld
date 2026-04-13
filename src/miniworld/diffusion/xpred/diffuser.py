"""VE x-prediction diffuser (Euclidean).

Noise process (same as EDM):  y = x + sigma * eps
Network predicts:             F_theta(c_in * y; sigma) = x0 / sigma_data
Preconditioning:              c_skip = 0, c_out = sigma_data  (recover x0 = F * sigma_data)
                              c_in   = 1/sqrt(sigma^2 + sigma_data^2)  (EDM)
Loss weight:                  (sigma + sigma_data)^2 / sigma^2  (v-loss in sigma space)
Loss target:                  x0 / sigma_data  (unit-scale target for numerical stability)
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float

from miniworld.diffusion.edm.diffuser import EuclideanDiffuser
from miniworld.diffusion.edm.scheduler import EDMScheduler
from miniworld.utils.structure.align import weighted_align


class XPredEuclideanDiffuser(EuclideanDiffuser):
    """VE x-prediction diffuser for Euclidean coordinates.

    The network predicts x0/sigma_data (unit-scale output).
    Interface is identical to ``EuclideanDiffuser``.
    """

    def __init__(
        self,
        config: EuclideanDiffuser.EuclideanConfig,
        scheduler: EDMScheduler,
    ) -> None:
        super().__init__(config, scheduler)

    @property
    def sigma_data(self) -> float:
        """Data standard deviation from scheduler config."""
        return self.scheduler.config.sigma_data

    max_loss_weight: float = 100.0

    def loss_weight(self, sigma: Float[torch.Tensor, ...]) -> Float[torch.Tensor, ...]:
        """v-loss weight in sigma space: clamp((sigma + sigma_data)^2 / sigma^2, max)."""
        w = (sigma + self.sigma_data) ** 2 / sigma**2
        return w.clamp(max=self.max_loss_weight)

    # -- forward (training) ---------------------------------------------------

    def sample(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        mask: Bool[torch.Tensor, "... L"] | None = None,
        num_augment: int = 1,
    ) -> tuple[
        Float[torch.Tensor, "... L 3"],
        Float[torch.Tensor, "... L 3"],
        Bool[torch.Tensor, "... L"] | None,
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
    ]:
        """VE noise: y = x + sigma*eps, input = c_in*y."""
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)

        batch_size = x0.shape[0]
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)

        if len(x0.shape) == 3:
            x0 = x0.expand(num_augment, *x0.shape[1:])
            if mask is not None:
                mask = mask.expand(num_augment, *mask.shape[1:])
        elif len(x0.shape) == 4:
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            x0 = x0.reshape(-1, *x0.shape[2:]).repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:]).repeat(num_expand, 1)

        x0 = self.random_rotation_and_translation(x0)

        total_num = x0.shape[0]
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        sigma = self.scheduler.sample_noise(total_num)
        sigma = sigma.view(sigma_shape).to(device=device, dtype=dtype)
        noise = torch.randn_like(x0)
        y = x0 + sigma * noise
        x_input = y * self.scheduler.input_scale(sigma)
        t_emb = self.scheduler.noise_condition(sigma).to(device=device, dtype=dtype)

        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma = sigma.view(num_augment, batch_size, *sigma.shape[1:])
        x_input = x_input.view(num_augment, batch_size, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, batch_size, *t_emb.shape[1:])

        return x0, x_input, mask, t_emb, sigma

    # -- loss -----------------------------------------------------------------

    def cal_loss(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],  # noqa: ARG002
        x_update: Float[torch.Tensor, "... L 3"],
        sigma: Float[torch.Tensor, ...],
        mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> Float[torch.Tensor, 1]:
        """Loss on normalized target: lambda * ||F_theta - x0/sigma_data||^2."""
        sd = self.sigma_data
        dtype = x_update.dtype
        x0 = x0.to(dtype=dtype)

        if mask is None:
            mask = torch.ones(x0.shape[:-1], device=x0.device, dtype=torch.bool)

        weight = self.loss_weight(sigma).to(dtype=dtype)
        weight = weight * mask.unsqueeze(-1)

        x_pred_orig = x_update * sd
        x0_aligned = weighted_align(x0, x_pred_orig, weight=mask.to(dtype=dtype))
        per_sample_loss = (
            (x_update - x0_aligned / sd).pow(2) * weight
        ).sum(dim=(-2, -1))
        n_valid = mask.sum(dim=-1).clamp(min=1) * 3
        return (per_sample_loss / n_valid).mean()
