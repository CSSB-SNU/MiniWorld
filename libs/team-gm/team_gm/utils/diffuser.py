import torch
import numpy as np
from scipy.spatial.transform import Rotation
from abc import ABC, abstractmethod
from typing import TypeVar, Any
from team_gm.utils.scheduler import DiffusionScheduler
# from team_gm.utils.losses import cal_smooth_lddt_loss

from pydantic import BaseModel

schedulerT = TypeVar("T", bound="DiffusionScheduler")


# TODO move this function


@torch.no_grad()
def weighted_align(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Align x to y using weighted least squares.
    """
    assert x.shape == y.shape, "x and y must have the same shape."
    assert weight.shape == x.shape[:-1], (
        "weight must have the same shape as x and y except for the last dimension."
    )
    assert x.ndim >= 2, "x and y must have at least 2 dimensions."
    assert x.shape[-1] == 3, "Last dimension of x and y must be of size 3."

    L = x.shape[-2]  # Length of the sequence
    original_shape = x.shape
    x, y = x.reshape(-1, L, 3), y.reshape(-1, L, 3)  # (AB, L, 3)
    weight = weight.reshape(-1, L)  # (AB, L)

    if weight is None:
        weight = torch.ones_like(x[..., 0])

    # Compute the weighted centroids
    w_sum = weight.sum(dim=-1, keepdim=True)
    weight = weight.unsqueeze(-1)  # (AB, L, 1)
    x_centroid = (x * weight).sum(dim=-2) / w_sum
    y_centroid = (y * weight).sum(dim=-2) / w_sum

    x_centroid = x_centroid.unsqueeze(-2)  # (AB, 1, 3)
    y_centroid = y_centroid.unsqueeze(-2)  # (AB, 1, 3)

    # Center the points
    x_centered = x - x_centroid
    y_centered = y - y_centroid

    # Compute the covariance matrix
    cov_matrix = torch.einsum("bni,bnj->bij", x_centered * weight, y_centered)

    # Singular Value Decomposition
    u, s, v = torch.linalg.svd(cov_matrix)
    v = v.mH

    # rotation_matrix = torch.einsum("bij,bjk -> bik", u, v)
    rotation_matrix = torch.einsum("bij,bkj -> bik", u, v)
    F = torch.eye(3, dtype=cov_matrix.dtype, device=cov_matrix.device)[None].repeat(
        x.shape[0], 1, 1
    )
    F[:, -1, -1] = torch.where(
        torch.det(rotation_matrix) < 0,
        torch.tensor(-1.0, dtype=rotation_matrix.dtype, device=rotation_matrix.device),
        torch.tensor(1.0, dtype=rotation_matrix.dtype, device=rotation_matrix.device),
    )
    # rotation_matrix = torch.einsum(
    #     "bij, bjk, bkl -> bil",
    #     u,
    #     F,
    #     v,
    # )
    rotation_matrix = torch.einsum(
        "bij, bjk, blk -> bil",
        u,
        F,
        v,
    )

    aligned_x = torch.einsum("bni, bij -> bnj", x_centered, rotation_matrix) + y_centroid
    # aligned_x = torch.einsum("bni, bji -> bnj", x_centered, rotation_matrix) + y_centroid

    # restore original shape
    aligned_x = aligned_x.reshape(*original_shape)

    return aligned_x


class Diffuser(ABC):
    """Base class for defining a diffusion model. (use solver when sampling)"""

    class DiffuserConfig(BaseModel):
        """Configuration for the Diffuser class."""

        method: str = "EDM"
        seed: int = 0
        translation_noise: float = 1.0
        # Add any additional configuration parameters here

    def __init__(
        self,
        config: DiffuserConfig,
        scheduler: schedulerT,
    ):
        self.config = config
        self.scheduler = scheduler
        self.clear_buffer()
        self._set_seed(config.seed)

    def _set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def clear_buffer(self):
        """Clear all internal buffers."""
        self._buffer = {}

    def assert_empty_buffer(self):
        """Assert that the internal buffer is empty."""
        if self._buffer:
            raise AssertionError(
                "Buffer is not empty. Please clear the buffer before using the diffuser."
            )

    @torch.no_grad()
    def random_rotation_and_translation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation and translation to the input tensor.
        """
        assert x.ndim >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == 3, "Last dimension of input tensor must be of size 3."
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (AB, L, 3) or (B, L, 3)

        # random rotation matrix
        n = x.shape[0]
        rot_mats = torch.from_numpy(Rotation.random(n).as_matrix()).to(x.device, x.dtype)

        # random translation vector
        translation = (
            torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
            * self.config.translation_noise
        )

        # Apply rotation and translation
        x = torch.bmm(x, rot_mats.transpose(-1, -2))  # -> (n, L, 3)
        x = x + translation  # (AB, L, 3)
        x = x.reshape(*x_shape)  # Restore original shape
        return x

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def cal_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute loss between model output and ground truth."""
        pass


class EuclideanDiffuser(Diffuser, ABC):
    class EuclideanConfig(BaseModel):
        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0

    def __init__(
        self,
        config: EuclideanConfig,
        scheduler: schedulerT,
    ):
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32
        self.clear_buffer()

    def sample(
        self, x0: torch.Tensor, mask: torch.Tensor | None, num_augment: int = 1
    ) -> torch.Tensor:
        """
        Add noise to atom_pos and store preconditioning data.
        """
        assert num_augment >= 1, "num_augment must be at least 1"

        B = x0.shape[0]
        self.clear_buffer()
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)
        x0 = x0.expand(num_augment, *x0.shape[1:])
        x0 = self.random_rotation_and_translation(x0)

        # random rotation and translation augmentation

        if mask is not None:
            mask = mask.expand(num_augment, *mask.shape[1:])
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
            }
        )

        return x_input, t_emb

    def cal_loss(self, x_update: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss between model prediction and true signal."""
        assert x_update.shape == self._buffer["noisy_x"].shape, (
            "x_update shape must match noisy_x shape in the buffer."
        )
        assert x_update.dtype == self.dtype, (
            "x_update must be of type float32, but got dtype: " + str(x_update.dtype)
        )
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
        # x0_aligned = x0
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
            raise ValueError("NaN detected in the loss calculation.")
        mse_loss = ((x_pred - x0_aligned).pow(2) * weight).mean()
        # mse_loss_before_align = ((x_pred - x0).pow(2) * weight).mean()
        # TODO bond loss

        # smooth lddt loss using checkpointing
        # smooth_lddt_loss = torch.utils.checkpoint.checkpoint(
        #     cal_smooth_lddt_loss,
        #     x_pred,
        #     x0,
        #     mask,
        #     use_reentrant=False,
        # )

        # loss = mse_loss + smooth_lddt_loss
        loss = mse_loss

        return loss


if __name__ == "__main__":
    # test align
    B = 2
    n = 10
    y = torch.randn(B, n, 3)

    # random rotation and translation
    rot_mats = torch.from_numpy(Rotation.random(B).as_matrix()).to(y.device, y.dtype)
    x = torch.einsum("bij,bjk->bik", y, rot_mats)  # (B, n, 3)
    translation = torch.randn(B, 1, 3, device=y.device, dtype=y.dtype)
    x = x + translation  # (2, n, 3)

    x_aligned = weighted_align(x, y)
    print(f"x_aligned-y: {torch.norm(x_aligned - y)}")
    breakpoint()
