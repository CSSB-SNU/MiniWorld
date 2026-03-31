from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from miniworld.diffusion.scheduler import DiffusionScheduler, EDMScheduler
from miniworld.utils.structure.align import weighted_align
from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid

from .scheduler import DecoupledEDMScheduler


class Diffuser(ABC):
    """Base class for defining a diffusion model. (use solver when sampling)."""

    class DiffuserConfig(BaseModel):
        """Configuration for the Diffuser class."""

        method: str = "EDM"
        seed: int = 0
        translation_noise: float = 1.0
        # Add any additional configuration parameters here

    def __init__(
        self,
        config: DiffuserConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self.clear_buffer()
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def clear_buffer(self) -> None:
        """Clear all internal buffers."""
        self._buffer = {}

    @torch.no_grad()
    def random_rotation_and_translation(
        self,
        x: Float[torch.Tensor, "... L 3"],
    ) -> Float[torch.Tensor, "... L 3"]:
        """Apply random rotation and translation to the input tensor."""
        if x.ndim < 2:
            msg = "Input tensor must have at least 2 dimensions."
            raise ValueError(msg)
        if x.shape[-1] != 3:
            msg = "Last dimension of input tensor must be of size 3."
            raise ValueError(msg)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (AB, L, 3) or (B, L, 3)

        n = x.shape[0]
        rot_mats = torch.from_numpy(Rotation.random(n).as_matrix()).to(
            x.device,
            x.dtype,
        )
        translation = (
            torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
            * self.config.translation_noise
        )

        x = torch.bmm(x, rot_mats.transpose(-1, -2))
        x = x + translation
        return x.reshape(*x_shape)

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample noisy input and store preconditioning data."""

    @abstractmethod
    def cal_loss(self, *args: Any, **kwargs: Any) -> Float[torch.Tensor, 1]:
        """Compute loss between model output and ground truth."""


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


class DecoupledEDMDiffuser(Diffuser):
    """Diffuser class for the decoupled EDM diffusion process."""

    scheduler: DecoupledEDMScheduler

    class DecoupledConfig(BaseModel):
        """Configuration for the DecoupledEDMDiffuser class."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 0.0

    def __init__(
        self,
        config: DecoupledConfig,
        scheduler: DecoupledEDMScheduler,
    ) -> None:
        self.config = config
        self.scheduler: DecoupledEDMScheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32
        self.clear_buffer()

    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
        atom_chain_break: dict[str, tuple[int, int]] | None,
        num_augment: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to batch.atom_pos and store preconditioning data.

        For now, this assumes a shared atom_chain_break mapping across the batch.
        """
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)
        if atom_chain_break is None:
            msg = "atom_chain_break must be provided for decoupled EDM diffusion."
            raise ValueError(msg)

        batch_size = x0.shape[0]
        self.clear_buffer()
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

        total_num = x0.shape[0]
        chain_num = len(atom_chain_break)
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        sigma_y, sigma_rotation, sigma_translation = self.scheduler.sample_noise(
            total_num,
            uniform=True,
        )
        sigma_y = sigma_y.to(device=device, dtype=dtype)
        sigma_rotation = sigma_rotation.to(device=device, dtype=dtype)
        sigma_translation = sigma_translation.to(device=device, dtype=dtype)

        noise = torch.randn_like(x0, device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma_y.view(sigma_shape)

        R, T = sample_rigid(
            sigma_rotation,
            sigma_translation,
            C=chain_num,
            device=device,
            dtype=dtype,
        )
        noisy_x = apply_chain_rt(noisy_x, R, T, atom_chain_break)

        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma_y = sigma_y.view(num_augment, batch_size, *sigma_y.shape[1:])
        sigma_rotation = sigma_rotation.view(
            num_augment,
            batch_size,
            *sigma_rotation.shape[1:],
        )
        sigma_translation = sigma_translation.view(
            num_augment,
            batch_size,
            *sigma_translation.shape[1:],
        )
        noisy_x = noisy_x.view(num_augment, batch_size, *noisy_x.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])

        self._buffer.update(
            {
                "x0": x0,
                "R": R,
                "T": T,
                "atom_chain_break": atom_chain_break,
                "sigma_y": sigma_y,
                "sigma_translation": sigma_translation,
                "noisy_x": noisy_x,
                "mask": mask,
            },
        )

        return noisy_x, sigma_y, sigma_rotation, sigma_translation

    def cal_loss(self, x_update: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss between model prediction and true signal."""
        if x_update.shape != self._buffer["noisy_x"].shape:
            msg = "x_update shape must match noisy_x shape in the buffer."
            raise ValueError(msg)
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype,
            )
            raise ValueError(msg)

        x0 = self._buffer["x0"]
        R, T = self._buffer["R"], self._buffer["T"]
        atom_chain_break = self._buffer["atom_chain_break"]
        sigma_y = self._buffer["sigma_y"]
        sigma_translation = self._buffer["sigma_translation"]
        noisy_x = self._buffer["noisy_x"]
        mask = self._buffer["mask"]

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype).reshape(-1, *x0.shape[-2:])
        noisy_x = noisy_x.to(dtype=dtype).reshape(-1, *noisy_x.shape[-2:])
        x_update = x_update.reshape(-1, *x_update.shape[-2:])
        sigma_y = sigma_y.to(dtype=dtype).reshape(-1)
        sigma_translation = sigma_translation.to(dtype=dtype).reshape(-1)

        c_skip = self.scheduler.skip_scale(sigma_y).to(dtype=dtype).view(-1, 1, 1)
        c_out = self.scheduler.output_scale(sigma_y).to(dtype=dtype).view(-1, 1, 1)
        weight = self.scheduler.loss_weight(sigma_y).to(dtype=dtype).view(-1, 1, 1)

        if mask is None:
            mask = torch.ones(
                x0.shape[:-1],
                device=x0.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.reshape(-1, *mask.shape[-1:])
            weight = weight * mask.unsqueeze(-1)

        noisy_x = apply_chain_rt(noisy_x, R, T, atom_chain_break, inverse=True)
        noisy_x = torch.where(mask.unsqueeze(-1), noisy_x, torch.zeros_like(noisy_x))
        x0 = torch.where(mask.unsqueeze(-1), x0, torch.zeros_like(x0))

        x_pred = c_skip * noisy_x + c_out * x_update
        x0_aligned = weighted_align(x0, x_pred, weight=mask.to(dtype=dtype))
        if torch.isnan((x_pred - x0_aligned).pow(2).mean()):
            torch.save(
                {
                    "sigma_y": sigma_y,
                    "sigma_translation": sigma_translation,
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
