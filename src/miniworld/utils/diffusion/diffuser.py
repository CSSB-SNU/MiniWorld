from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from miniworld.utils.diffusion.scheduler import DecoupledEDMScheduler
from miniworld.utils.structure.align import weighted_align
from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid

if TYPE_CHECKING:
    from miniworld.utils.diffusion.scheduler import DiffusionScheduler

SchedulerT = TypeVar("SchedulerT", bound="DiffusionScheduler")


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
        scheduler: SchedulerT,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self.clear_buffer()
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        np.random.seed(seed)  # noqa: NPY002
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

    def assert_empty_buffer(self) -> None:
        """Assert that the internal buffer is empty."""
        if self._buffer:
            msg = "Buffer is not empty. Please clear the buffer before using the diffuser."
            raise AssertionError(msg)

    @torch.no_grad()
    def random_rotation_and_translation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation and translation to the input tensor."""
        if x.ndim < 2:
            msg = "Input tensor must have at least 2 dimensions."
            raise ValueError(msg)
        if x.shape[-1] != 3:
            msg = "Last dimension of input tensor must be of size 3."
            raise ValueError(msg)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (AB, L, 3) or (B, L, 3)

        # random rotation matrix
        n = x.shape[0]
        rot_mats = torch.from_numpy(Rotation.random(n).as_matrix()).to(
            x.device, x.dtype
        )

        # random translation vector
        translation = (
            torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
            * self.config.translation_noise
        )

        # Apply rotation and translation
        x = torch.bmm(x, rot_mats.transpose(-1, -2))  # -> (n, L, 3)
        x = x + translation  # (AB, L, 3)
        return x.reshape(*x_shape)  # Restore original shape

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample noisy input and store preconditioning data."""

    @abstractmethod
    def cal_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute loss between model output and ground truth."""


class EuclideanDiffuser(Diffuser, ABC):
    """Diffuser class for Euclidean diffusion process."""

    class EuclideanConfig(BaseModel):
        """Configuration for the EuclideanDiffuser class."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0

    def __init__(
        self,
        config: EuclideanConfig,
        scheduler: SchedulerT,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32
        self.clear_buffer()

    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
        num_augment: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Add noise to batch.atom_pos and store preconditioning data."""
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)

        B = x0.shape[0]
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

        # random rotation and translation augmentation
        AB = x0.shape[0]
        sigma_shape = (AB,) + (1,) * (x0.ndim - 1)

        sigma = self.scheduler.sample_noise(AB)
        noise = torch.randn_like(x0, device=device, dtype=dtype)
        sigma = sigma.view(sigma_shape).to(device=device, dtype=dtype)
        input_scaling = self.scheduler.input_scale(sigma).to(device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma
        x_input = noisy_x * input_scaling
        t_emb = self.scheduler.noise_condition(sigma).to(device=device, dtype=dtype)

        x0 = x0.view(num_augment, B, *x0.shape[1:])
        sigma = sigma.view(num_augment, B, *sigma.shape[1:])
        noisy_x = noisy_x.view(num_augment, B, *noisy_x.shape[1:])
        x_input = x_input.view(num_augment, B, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, B, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, B, *t_emb.shape[1:])

        self._buffer.update(
            {
                "x0": x0,
                "sigma": sigma,
                "noisy_x": noisy_x,
                "mask": mask,
            },
        )

        return x_input, mask, t_emb

    def cal_loss(self, x_update: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss between model prediction and true signal."""
        if x_update.shape != self._buffer["noisy_x"].shape:
            msg = "x_update shape must match noisy_x shape in the buffer."
            raise ValueError(msg)
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype
            )
            raise ValueError(msg)
        sigma = self._buffer["sigma"]
        noisy_x = self._buffer["noisy_x"]
        x0 = self._buffer["x0"]
        mask = self._buffer["mask"]

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype)
        noisy_x = noisy_x.to(dtype=dtype)
        c_skip = self.scheduler.skip_scale(sigma).to(dtype=dtype)
        c_out = self.scheduler.output_scale(sigma).to(dtype=dtype)
        weight = self.scheduler.loss_weight(sigma).to(dtype=dtype)
        if mask is not None:
            weight = weight * mask.unsqueeze(-1)

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
    """Diffuser class for decoupled EDM diffusion process."""

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
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32
        self.clear_buffer()

    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        atom_chain_break: dict[str, int] | None,
        num_augment: int = 1,
    ) -> torch.Tensor:
        """Add noise to batch.atom_pos and store preconditioning data.

        for now, we assume B = 1 (if not, we have to handle list of atom_chain_break.)
        """
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)

        B = x0.shape[0]
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

        x0 = self.random_rotation_and_translation(x0)  # (AB, L, 3)

        # random rotation and translation augmentation
        AB = x0.shape[0]
        C = len(atom_chain_break)
        sigma_shape = (AB,) + (1,) * (x0.ndim - 1)

        sigma_y, sigma_R, sigma_T = self.scheduler.sample_noise(AB, uniform=True)
        sigma_y = sigma_y.to(device=device, dtype=dtype)
        sigma_R = sigma_R.to(device=device, dtype=dtype)
        sigma_T = sigma_T.to(device=device, dtype=dtype)

        noise = torch.randn_like(x0, device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma_y.view(sigma_shape)

        # apply SE(3)
        R, T = sample_rigid(sigma_R, sigma_T, C)
        noisy_x = apply_chain_rt(noisy_x, R, T, atom_chain_break)

        input_scaling = self.scheduler.input_scale(sigma_y, sigma_T).to(
            device=device, dtype=dtype
        )
        t_emb = self.scheduler.noise_condition(sigma_y).to(
            device=device, dtype=dtype
        )  # follow sigma_y
        x_input = noisy_x * input_scaling.view(sigma_shape)

        x0 = x0.view(num_augment, B, *x0.shape[1:])
        sigma_y = sigma_y.view(num_augment, B, *sigma_y.shape[1:])
        noisy_x = noisy_x.view(num_augment, B, *noisy_x.shape[1:])
        x_input = x_input.view(num_augment, B, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, B, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, B, *t_emb.shape[1:])

        self._buffer.update(
            {
                "x0": x0,
                "R": R,
                "T": T,
                "atom_chain_break": atom_chain_break,
                "sigma_y": sigma_y,
                "sigma_T": sigma_T,
                "noisy_x": noisy_x,
                "mask": mask,
            },
        )

        return noisy_x, sigma_y, sigma_R, sigma_T

    def cal_loss(self, x_update: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss between model prediction and true signal."""
        if x_update.shape != self._buffer["noisy_x"].shape:
            msg = "x_update shape must match noisy_x shape in the buffer."
            raise ValueError(msg)
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype
            )
            raise ValueError(msg)
        x0 = self._buffer["x0"]
        R, T = self._buffer["R"], self._buffer["T"]
        atom_chain_break = self._buffer["atom_chain_break"]
        sigma_y = self._buffer["sigma_y"]
        sigma_T = self._buffer["sigma_T"]
        noisy_x = self._buffer["noisy_x"]
        mask = self._buffer["mask"]

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype)
        noisy_x = noisy_x.to(dtype=dtype)
        c_skip = self.scheduler.skip_scale(sigma_y).to(dtype=dtype)
        c_out = self.scheduler.output_scale(sigma_y).to(dtype=dtype)
        weight = self.scheduler.loss_weight(sigma_y).to(dtype=dtype)
        if mask is not None:
            weight = weight * mask.unsqueeze(-1)

        # apply SE(3) inverse transform
        noisy_x = apply_chain_rt(noisy_x, R, T, atom_chain_break, inverse=True)
        noisy_x = torch.where(mask.unsqueeze(-1), noisy_x, torch.zeros_like(x0))
        x0 = torch.where(mask.unsqueeze(-1), x0, torch.zeros_like(x0))

        x_pred = c_skip * noisy_x + c_out * x_update

        # align x0 to x_pred
        x0_aligned = weighted_align(x0, x_pred, weight=mask.to(dtype=dtype))
        if torch.isnan((x_pred - x0_aligned).pow(2).mean()):
            torch.save(
                {
                    "sigma_y": sigma_y,
                    "sigma_T": sigma_T,
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
        diff = x_pred - x0_aligned
        return (diff.pow(2) * weight).mean()
