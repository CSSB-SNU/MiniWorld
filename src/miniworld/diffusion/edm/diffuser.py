"""EDM Euclidean diffuser."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel

from miniworld.diffusion.base.diffuser import Diffuser
from miniworld.diffusion.edm.scheduler import EDMScheduler
from miniworld.utils.structure.align import weighted_align

if TYPE_CHECKING:
    from miniworld.diffusion.base.scheduler import DiffusionScheduler


# ruff: noqa: PLR2004
class EuclideanDiffuser(Diffuser, ABC):
    """Diffuser class for Euclidean diffusion process."""

    scheduler: EDMScheduler

    class EuclideanConfig(BaseModel):
        """Configuration for the EuclideanDiffuser class."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0

    def __init__(
        self,
        config: EuclideanConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        if not isinstance(scheduler, EDMScheduler):
            msg = "EuclideanDiffuser requires an EDMScheduler-compatible scheduler."
            raise TypeError(msg)
        self.config = config
        self.scheduler: EDMScheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32

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
        """Add noise to batch.atom_pos and store preconditioning data."""
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)

        batch_size = x0.shape[0]
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)
        if len(x0.shape) == 3:  # x0 : (B, L, 3)
            x0 = x0.expand(num_augment, *x0.shape[1:])
            if mask is not None:
                mask = mask.expand(num_augment, *mask.shape[1:])
        elif len(x0.shape) == 4:  # x0 : (B, N_str, L, 3)
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            x0 = x0.reshape(-1, *x0.shape[2:])
            x0 = x0.repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:])
                mask = mask.repeat(num_expand, 1)

        x0 = self.random_rotation_and_translation(x0)

        # random rotation and translation augmentation
        total_num = x0.shape[0]
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        sigma = self.scheduler.sample_noise(total_num)
        noise = torch.randn_like(x0, device=device, dtype=dtype)
        sigma = sigma.view(sigma_shape).to(device=device, dtype=dtype)
        input_scaling = self.scheduler.input_scale(sigma).to(device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma
        x_input = noisy_x * input_scaling
        t_emb = self.scheduler.noise_condition(sigma).to(device=device, dtype=dtype)

        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma = sigma.view(num_augment, batch_size, *sigma.shape[1:])
        noisy_x = noisy_x.view(num_augment, batch_size, *noisy_x.shape[1:])
        x_input = x_input.view(num_augment, batch_size, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, batch_size, *t_emb.shape[1:])

        return x0, x_input, mask, t_emb, sigma

    def cal_loss(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        x_update: Float[torch.Tensor, "... L 3"],
        sigma: Float[torch.Tensor, ...],
        mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> Float[torch.Tensor, 1]:
        """Compute EDM loss between model prediction and true signal."""
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype,
            )
            raise ValueError(msg)
        input_scaling = self.scheduler.input_scale(sigma)
        input_scaling = input_scaling.to(device=x0.device, dtype=self.dtype)
        noisy_x = x_input / input_scaling

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype)
        noisy_x = noisy_x.to(dtype=dtype)
        c_skip = self.scheduler.skip_scale(sigma).to(dtype=dtype)
        c_out = self.scheduler.output_scale(sigma).to(dtype=dtype)
        weight = self.scheduler.loss_weight(sigma).to(dtype=dtype)
        if mask is not None:
            weight = weight * mask.unsqueeze(-1)
        else:
            mask = torch.ones(
                x0.shape[:-1],
                device=x0.device,
                dtype=torch.bool,
            )

        x_pred = c_skip * noisy_x + c_out * x_update
        # align x0 to x_pred
        x0_aligned = weighted_align(x0, x_pred, weight=mask.to(dtype=dtype))
        if torch.isnan((x_pred - x0_aligned).pow(2).mean()):
            torch.save(
                {
                    "sigma": sigma,
                    "c_skip": c_skip,
                    "c_out": c_out,
                    "x_update": x_update,
                    "x_pred": x_pred,
                    "x0_aligned": x0_aligned,
                    "x0": x0,
                    "noisy_x": noisy_x,
                    "mask": mask,
                    "weight": weight,
                },
                "debug_nan_at_loss.pt",
            )
            msg = "NaN detected in the loss calculation."
            raise ValueError(msg)
        return ((x_pred - x0_aligned).pow(2) * weight).mean()


