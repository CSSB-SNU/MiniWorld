import torch
import numpy as np
from team_gm.utils.diffuser import Diffuser, weighted_align
from MiniWorld.utils.scheduler import DecoupledEDMScheduler
from MiniWorld.utils.se3 import apply_chain_rt, sample_rigid
# from team_gm.utils.losses import cal_smooth_lddt_loss

from pydantic import BaseModel



class DecoupledEDMDiffuser(Diffuser):
    class DecoupledConfig(BaseModel):
        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 0.0

    def __init__(
        self,
        config: DecoupledConfig,
        scheduler: DecoupledEDMScheduler,
    ):
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32
        self.clear_buffer()



    def sample(
        self, x0: torch.Tensor, mask: torch.Tensor, atom_chain_break : dict[str, int] | None, num_augment: int = 1
    ) -> torch.Tensor:
        """
        Add noise to batch.atom_pos and store preconditioning data.

        for now, we assume B = 1 (if not, we have to handle list of atom_chain_break.)
        """
        assert num_augment >= 1, "num_augment must be at least 1"

        B = x0.shape[0]
        self.clear_buffer()
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)
        if len(x0.shape) == 3: # x0 : (B, L, 3)
            x0 = x0.expand(num_augment, *x0.shape[1:])
            if mask is not None:
                mask = mask.expand(num_augment, *mask.shape[1:])
        elif len(x0.shape) == 4: # x0 : (B, N_str, L, 3)
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            x0 = x0.reshape(-1, *x0.shape[2:])
            x0 = x0.repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:])
                mask = mask.repeat(num_expand, 1)

        x0 = self.random_rotation_and_translation(x0) # (AB, L, 3)

        # random rotation and translation augmentation
        AB = x0.shape[0]
        C = len(atom_chain_break)
        sigma_shape = (AB,) + (1,) * (x0.ndim - 1)

        sigma_y, sigma_R, sigma_T = self.scheduler.sample_noise(AB, uniform = True)
        sigma_y = sigma_y.to(device=device, dtype=dtype)
        sigma_R = sigma_R.to(device=device, dtype=dtype)
        sigma_T = sigma_T.to(device=device, dtype=dtype)

        noise = torch.randn_like(x0, device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma_y.view(sigma_shape)

        # apply SE(3)
        R, T = sample_rigid(sigma_R, sigma_T, C)
        noisy_x = apply_chain_rt(noisy_x, R, T, atom_chain_break)

        input_scaling = self.scheduler.input_scale(sigma_y, sigma_T).to(device=device, dtype=dtype)
        t_emb = self.scheduler.noise_condition(sigma_y).to(device=device, dtype=dtype) # follow sigma_y
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
            }
        )

        # return x_input, t_emb
        return noisy_x, sigma_y, sigma_R, sigma_T

    def cal_loss(self, x_update: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss between model prediction and true signal."""
        assert x_update.shape == self._buffer["noisy_x"].shape, (
            "x_update shape must match noisy_x shape in the buffer."
        )
        assert x_update.dtype == self.dtype, (
            "x_update must be of type float32, but got dtype: " + str(x_update.dtype)
        )
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

        x_pred = c_skip * noisy_x + c_out * x_update
        # align x0 to x_pred
        x0_aligned = weighted_align(x0, x_pred, weight=mask.to(dtype=dtype))
        # x0_aligned = x0
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
            raise ValueError("NaN detected in the loss calculation.")
        mse_loss = ((x_pred - x0_aligned).pow(2) * weight).mean()
        loss = mse_loss

        return loss
