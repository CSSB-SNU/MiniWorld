from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from team_gm.utils.diffusion import Diffuser

from .discrete_scheduler import D3PMScheduler, SEDDScheduler
from .util import symmetrize_labels, symmetrize_pair

if TYPE_CHECKING:
    from team_gm.utils.diffusion.scheduler import DiffusionScheduler


class SEDDSampleOutput(BaseModel):
    """Output of one SEDD diffusion sampling step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_ratio: torch.Tensor
    xt_one_hot: torch.Tensor
    mask: torch.Tensor | None
    t_emb: torch.Tensor
    sigma: torch.Tensor

class SEDDDiffuser(Diffuser):
    """SEDD diffuser using score-entropy loss on discrete pairwise labels."""

    class SEDDConfig(BaseModel):
        """Configuration for the SEDDDiffuser class."""

        method: str = "SEDD"
        seed: int = 0
        enforce_symmetric: bool = True
        weight_with_sigma: bool = True
        min_ratio: float = 1e-5

    def __init__(
        self,
        config: SEDDConfig,
        scheduler: "DiffusionScheduler",
    ) -> None:
        if not isinstance(scheduler, SEDDScheduler):
            msg = "SEDDDiffuser requires an SEDDScheduler."
            raise TypeError(msg)
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32

    def _prepare_inputs(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = x0.device
        if x0.dtype not in (torch.int64, torch.int32):
            x0 = x0.long()
        if len(x0.shape) not in {3, 4}:
            msg = f"SEDDDiffuser expects pairwise label tensors, got shape {x0.shape}"
            raise ValueError(msg)
        x0 = x0.to(device=device)
        if mask is not None:
            mask = mask.to(device=device)
            if mask.shape != x0.shape:
                msg = f"Mask shape {mask.shape} must match x0 shape {x0.shape}."
                raise ValueError(msg)
            mask = mask.bool()
        return x0, mask

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> SEDDSampleOutput:
        """Sample xt ~ p_t|0 and store targets for score-entropy loss."""
        x0, mask = self._prepare_inputs(x0, mask)
        device = x0.device

        B = x0.shape[0]
        t = self.scheduler.sample_noise(B).to(device)
        probs = self.scheduler.forward_prob(x0, t)

        flat_prob = probs.view(probs.shape[0], -1, probs.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt = sampled.view(*x0.shape)
        if self.config.enforce_symmetric:
            xt = symmetrize_labels(xt.clone())
            x0 = symmetrize_labels(x0.clone())
        xt_one_hot = F.one_hot(xt, num_classes=self.scheduler.num_classes).to(self.dtype)
        if self.config.enforce_symmetric:
            probs = symmetrize_pair(probs)

        # Target ratios p_t(y|x0) / p_t(xt|x0)
        denom = torch.gather(probs, dim=-1, index=xt.unsqueeze(-1)).clamp_min(
            self.config.min_ratio,
        )
        target_ratio = probs / denom
        target_ratio = target_ratio * (1.0 - xt_one_hot)

        sigma = self.scheduler.sigma(t).to(device)
        t_emb = self.scheduler.noise_condition(sigma).to(
            device=device,
            dtype=self.dtype,
        )

        return SEDDSampleOutput(
            target_ratio=target_ratio,
            xt_one_hot=xt_one_hot,
            mask=mask,
            t_emb=t_emb,
            sigma=sigma,
        )

    def cal_loss(
        self,
        sample_output: SEDDSampleOutput,
        ratio_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Score-entropy loss: sum_y≠x [s - r log s] with optional sigma weight."""
        target_ratio = sample_output.target_ratio
        xt_one_hot = sample_output.xt_one_hot
        mask = sample_output.mask
        sigma = sample_output.sigma
        ratio_pred = ratio_pred.clamp_min(self.config.min_ratio)
        off_diag = 1.0 - xt_one_hot.to(dtype=ratio_pred.dtype)
        weight = off_diag
        if mask is not None:
            mask = mask.to(dtype=ratio_pred.dtype)
            weight = weight * mask.unsqueeze(-1)
        if self.config.weight_with_sigma:
            sigma = sigma.view((sigma.shape[0],) + (1,) * (weight.ndim - 1))
            weight = weight * sigma

        loss = (ratio_pred - target_ratio * torch.log(ratio_pred)) * weight
        denom = weight.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=ratio_pred.device, dtype=ratio_pred.dtype)
        return loss.sum() / denom

class D3PMSampleOutput(BaseModel):
    """Output of one D3PM diffusion sampling step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x0: torch.Tensor
    xt_one_hot: torch.Tensor
    mask: torch.Tensor | None
    t_emb: torch.Tensor

class D3PMDiffuser(Diffuser):
    """D3PM diffuser for discrete pairwise labels (e.g., contact/distance bins)."""

    class D3PMConfig(BaseModel):
        """Configuration for the D3PMDiffuser class."""

        method: str = "D3PM"
        seed: int = 0
        enforce_symmetric: bool = True

    def __init__(
        self,
        config: D3PMConfig,
        scheduler: "DiffusionScheduler",
    ) -> None:
        if not isinstance(scheduler, D3PMScheduler):
            msg = "D3PMDiffuser requires a D3PMScheduler."
            raise TypeError(msg)
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32

    def _prepare_inputs(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = x0.device
        if x0.dtype not in (torch.int64, torch.int32):
            x0 = x0.long()
        if len(x0.shape) not in {3, 4}:
            msg = f"D3PMDiffuser expects pairwise label tensors, got shape {x0.shape}"
            raise ValueError(msg)
        x0 = x0.to(device=device)
        if mask is not None:
            mask = mask.to(device=device)
            if mask.shape != x0.shape:
                msg = f"Mask shape {mask.shape} must match x0 shape {x0.shape}."
                raise ValueError(msg)
            mask = mask.bool()
        return x0, mask

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> D3PMSampleOutput:
        """Add discrete corruption to pairwise labels and return one-hot x_t."""
        x0, mask = self._prepare_inputs(x0, mask)
        device = x0.device
        B = x0.shape[0]
        sigma_shape = (B,) + (1,) * (x0.ndim - 1)

        timesteps = self.scheduler.sample_noise(B).to(device)
        alpha_bar = self.scheduler.alpha_bar(timesteps).to(device=device)

        one_hot = F.one_hot(x0, num_classes=self.scheduler.num_classes).to(self.dtype)
        alpha_term = alpha_bar.view((*sigma_shape, 1))
        if self.scheduler.config.transition_mode == "uniform":
            prob = alpha_term * one_hot + (1.0 - alpha_term) / float(
                self.scheduler.num_classes,
            )
        elif self.scheduler.config.transition_mode == "absorbing":
            mask_labels = torch.full_like(x0, self.scheduler.mask_class)
            mask_one_hot = F.one_hot(
                mask_labels,
                num_classes=self.scheduler.num_classes,
            ).to(self.dtype)
            prob = alpha_term * one_hot + (1.0 - alpha_term) * mask_one_hot
        else:
            msg = f"Unknown transition_mode {self.scheduler.config.transition_mode}"
            raise ValueError(msg)

        flat_prob = prob.view(prob.shape[0], -1, prob.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt = sampled.view(*x0.shape)
        if self.config.enforce_symmetric:
            xt = symmetrize_labels(xt.clone())
            x0 = symmetrize_labels(x0.clone())
        xt_one_hot = F.one_hot(xt, num_classes=self.scheduler.num_classes).to(self.dtype)
        if self.config.enforce_symmetric:
            one_hot = symmetrize_pair(one_hot)

        t_emb = self.scheduler.noise_condition(timesteps).to(
            device=device,
            dtype=self.dtype,
        )

        return D3PMSampleOutput(
            x0=x0,
            xt_one_hot=xt_one_hot,
            mask=mask,
            t_emb=t_emb,
        )


    def cal_loss(
        self,
        sample_output: D3PMSampleOutput,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss to original labels."""
        x0 = sample_output.x0
        mask = sample_output.mask
        if logits.shape[:-1] != x0.shape:
            msg = f"Logits shape {logits.shape} incompatible with x0 {x0.shape}"
            raise ValueError(msg)
        num_classes = logits.shape[-1]
        logits_flat = logits.view(-1, num_classes)
        x0_flat = x0.view(-1)
        loss = F.cross_entropy(logits_flat, x0_flat, reduction="none")
        loss = loss.view(x0.shape)
        if mask is not None:
            mask = mask.to(dtype=loss.dtype)
            loss = loss * mask
            denom = mask.sum()
            if denom <= 0:
                return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            return loss.sum() / denom
        return loss.mean()
