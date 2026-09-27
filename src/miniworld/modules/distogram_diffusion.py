"""EDM distogram diffusion where the TRUNK itself is the denoiser.

The distogram is treated as an L x L single-channel "image" whose pixel is the per-pair
distance BIN INDEX (0..D-1). Instead of a separate denoiser network, the trunk
(MiniMSAModule + MiniPairformer) IS the denoiser: the noised bin image is fed in through
a single Linear ENCODER (added to the trunk's pair input, in the slot recycling used to
re-inject the pair state), the trunk runs one pass, and a single Linear DECODER reads the
denoised bin image back out. EDM preconditioning (Karras et al. 2022) wraps it.

This module owns only the EDM machinery + two thin linear maps:
  * ``encoder``     : noised bin image (1 ch) -> pair rep addition  (into the trunk)
  * ``decoder``     : trunk output pair rep -> denoised bin image (1 ch)
The model orchestrates: encode_input -> run trunk -> decode_output.

The bin image is symmetric (``D_ij == D_ji``) with a masked diagonal, so noise is drawn
symmetrically and the decoded output is symmetrised. ``sigma_data`` / ``bin_center``
describe the scale / centre of the bin image. Calibrate them on the actual crop and
target using ``scripts/measure_distogram_sigma_data.py``; prior-run values are initial guesses.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, Field, model_validator
import torch.nn.functional as F
from team_gm import typecheck
from team_gm.diffusion import EDMScheduler
from team_gm.modules.layers.embeddings import fourier_embedding
from team_gm.modules.primitives import Linear
from torch import nn


class DistogramDiffusionConfig(BaseModel):
    """Config for the EDM distogram-diffusion head (trunk-as-denoiser)."""

    # Bin-image encoding: x0 = (bin - bin_center) (raw centred bins). None -> (D-1)/2.
    # sigma_data (in ``scheduler``) is the std of x0 measured on training data.
    bin_center: float | None = Field(default=None, allow_inf_nan=False)

    # Number of reverse (Heun) steps at inference — the compute/quality knob that
    # #recycles used to be.
    num_sampling_steps: int = Field(default=24, ge=2)

    # EDM noise schedule / preconditioning. sigma_data MUST be set from measurement.
    scheduler: EDMScheduler.EDMSchedulerConfig = Field(default_factory=EDMScheduler.EDMSchedulerConfig)

    @model_validator(mode="after")
    def validate_schedule(self):
        import math
        s = self.scheduler
        values = (s.sigma_data, s.sigma_min, s.sigma_max, s.rho, s.P_mean, s.P_std)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Diffusion schedule must be finite")
        if not (s.sigma_data > 0 and 0 < s.sigma_min < s.sigma_max and s.rho > 0 and s.P_std >= 0):
            raise ValueError("Invalid diffusion schedule")
        return self


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    """(x + x^T) / 2 over the last two (L, L) axes."""
    return 0.5 * (x + x.transpose(-1, -2))


def _symmetric_noise(
    b: int, length: int, *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric [B, L, L] Gaussian: mirror the strict upper triangle so every
    off-diagonal entry is a single ``N(0, 1)`` draw (variance preserved, unlike
    averaging ``(n + n^T)/2`` which halves it)."""
    e = torch.randn(b, length, length, device=device, dtype=dtype)
    upper = e.triu(diagonal=1)
    return upper + upper.transpose(-1, -2)


class DistogramDiffusion(nn.Module):
    """EDM machinery + thin linear encoder/decoder; the TRUNK is the denoiser.

    The model calls, per training step: ``sample_sigma`` -> ``add_noise`` ->
    ``encode_input`` (add to the trunk's pair input) -> run the trunk once ->
    ``decode_output`` -> ``loss``.
    """

    def __init__(
        self,
        d_pair: int,
        num_distogram_bins: int,
        config: DistogramDiffusionConfig,
    ) -> None:
        super().__init__()
        if num_distogram_bins < 2:
            raise ValueError("Distogram diffusion requires at least two bins")
        self.config = config
        self.num_bins = num_distogram_bins
        self.bin_center = (
            config.bin_center
            if config.bin_center is not None
            else (num_distogram_bins - 1) / 2.0
        )
        self.scheduler = EDMScheduler(config.scheduler)

        self.d_fourier = int(fourier_embedding(torch.zeros(1)).shape[-1])
        # Thin linear maps into / out of the trunk's pair representation. bf16 to match
        # the trunk; the EDM preconditioning around them stays fp32. The ENCODER takes
        # the noised bin image AND the Fourier noise-level embedding together (concat) ->
        # d_pair, so the noise level is fed through the encoder rather than added apart.
        self.encoder = Linear(1 + self.d_fourier, d_pair, bias=False, init="default")
        self.decoder = Linear(d_pair, 1, bias=False, init="zero")
        self.to(torch.bfloat16)

    # -- bin <-> continuous image ------------------------------------------------
    def encode(self, bins: Int[torch.Tensor, "B L L"]) -> Float[torch.Tensor, "B L L"]:
        """Bin index -> centred continuous image x0 = bin - bin_center (fp32)."""
        return bins.float() - self.bin_center

    def decode(self, x0: Float[torch.Tensor, "B L L"]) -> Int[torch.Tensor, "B L L"]:
        """Continuous image -> nearest valid bin index, symmetrised + clamped."""
        bins = torch.round(_symmetrize(x0) + self.bin_center)
        return bins.clamp(0, self.num_bins - 1).long()

    # -- EDM noising -------------------------------------------------------------
    def sample_sigma(self, b: int, device: torch.device) -> torch.Tensor:
        """Device-local log-normal noise; CUDA graph replays draw fresh values."""
        s = self.config.scheduler
        n = b if s.use_time_augmentation else 1
        z = torch.randn(n, device=device, dtype=torch.float32)
        sigma = s.sigma_data * torch.exp(s.P_mean + s.P_std * z)
        return sigma.clamp(s.sigma_min * s.sigma_data, s.sigma_max * s.sigma_data).expand(b)

    @typecheck
    def add_noise(
        self,
        x0: Float[torch.Tensor, "B L L"],
        sigma: Float[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B L L"]:
        """x_t = x0 + sigma * symmetric_noise (keeps x_t symmetric)."""
        n = _symmetric_noise(
            x0.shape[0], x0.shape[1], device=x0.device, dtype=torch.float32,
        )
        return x0 + n * sigma[:, None, None]

    # -- encode into / decode out of the trunk (EDM preconditioned) --------------
    @typecheck
    def encode_input(
        self,
        x_t: Float[torch.Tensor, "B L L"],
        sigma: Float[torch.Tensor, "B"],
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "B L L C"]:
        """Noised bin image + noise level -> pair-rep addition for the trunk input.

        Applies the EDM input scaling ``c_in`` and adds the Fourier noise-level
        embedding (broadcast over the L x L grid).
        """
        c_in = self.scheduler.input_scale(sigma)[:, None, None]
        c_noise = self.scheduler.noise_condition(sigma)  # [B]
        x_scaled = (c_in * x_t)[..., None].to(dtype)  # [B, L, L, 1]
        fo = fourier_embedding(c_noise).to(dtype)  # [B, d_fourier]
        # Algebraically the same linear map as encoder(cat([x, Fourier(sigma)])).
        # Project the shared time embedding once, avoiding an L*L*Fourier tensor.
        image = F.linear(x_scaled, self.encoder.weight[:, :1])
        time = F.linear(fo, self.encoder.weight[:, 1:])
        return image + time[:, None, None, :]

    @typecheck
    def decode_output(
        self,
        token_pair: Float[torch.Tensor, "B L L C"],
        x_t: Float[torch.Tensor, "B L L"],
        sigma: Float[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B L L"]:
        """Trunk output -> denoised bin image, with EDM ``c_skip``/``c_out`` (fp32)."""
        f = _symmetrize(self.decoder(token_pair)[..., 0].float())  # [B, L, L]
        c_out = self.scheduler.output_scale(sigma)[:, None, None]
        c_skip = self.scheduler.skip_scale(sigma)[:, None, None]
        return c_skip * x_t + c_out * f

    # -- training loss -----------------------------------------------------------
    @typecheck
    def loss(
        self,
        x0_hat: Float[torch.Tensor, "B L L"],
        x0: Float[torch.Tensor, "B L L"],
        pair_mask: Bool[torch.Tensor, "B L L"],
        sigma: Float[torch.Tensor, "B"],
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
        interchain_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """EDM MSE, with v1.2's valid-pair denominator and interchain weighting."""
        if not 0 <= interchain_weight < float("inf"):
            raise ValueError("interchain_weight must be finite and nonnegative")
        if interchain_weight != 1.0 and token_asym_id is None:
            raise ValueError("Interchain weighting requires token_asym_id")
        if token_asym_id is not None and token_asym_id.shape != x0.shape[:-1]:
            raise ValueError("token_asym_id must match batch and token dimensions")
        length = x0.shape[1]
        upper = torch.ones(length, length, dtype=torch.bool, device=x0.device).triu(1)
        mask = pair_mask & upper
        w = self.scheduler.loss_weight(sigma)  # [B]

        error = torch.where(mask, x0_hat.float(), 0.) - torch.where(mask, x0.float(), 0.)
        sq = error.square()
        denom = mask.sum(dim=(-2, -1)).clamp_min(1).to(sq.dtype)
        weighted_sq = sq
        if interchain_weight != 1.0:
            cross = token_asym_id[:, :, None] != token_asym_id[:, None, :]
            weighted_sq = sq * torch.where(cross, interchain_weight, 1.0)
        per_sample = w * weighted_sq.sum(dim=(-2, -1)) / denom
        loss = per_sample.mean()

        with torch.no_grad():
            rmse = (sq.sum() / mask.sum().clamp_min(1)).sqrt()
        return loss, {
            "diffusion_loss": loss.detach(),
            "bin_rmse": rmse.detach(),
            "sigma_mean": sigma.mean().detach(),
        }

    # -- sampling (inference) ----------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        denoise_fn,  # (x_t[B,L,L], sigma[B]) -> x0_hat[B,L,L]
        b: int,
        length: int,
        device: torch.device,
        num_steps: int | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> Int[torch.Tensor, "B L L"]:
        """Heun (EDM deterministic) reverse process -> predicted bin indices.

        ``denoise_fn`` runs one trunk pass (encode_input -> trunk -> decode_output).
        """
        steps = self.config.num_sampling_steps if num_steps is None else num_steps
        if steps < 2:
            raise ValueError("Heun sampling requires at least two steps")
        # EDM Heun schedule ends at exactly zero (last step Euler). The shared
        # AF3 scheduler intentionally ends at sigma_min, so build this schedule here.
        s = self.config.scheduler
        ramp = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
        high, low = s.sigma_data * s.sigma_max, s.sigma_data * s.sigma_min
        levels = (high ** (1 / s.rho) + ramp * (low ** (1 / s.rho) - high ** (1 / s.rho))) ** s.rho
        sigmas = torch.cat((levels.float(), levels.new_zeros(1).float()))
        valid = ~torch.eye(length, device=device, dtype=torch.bool)[None]
        if token_mask is not None:
            valid = valid & token_mask[:, :, None] & token_mask[:, None, :]

        def _full(s: torch.Tensor) -> torch.Tensor:
            return s.to(torch.float32).expand(b)

        x = _symmetric_noise(b, length, device=device, dtype=torch.float32) * sigmas[0]
        x = x.masked_fill(~valid, 0)
        for i in range(steps):
            s_i, s_next = sigmas[i], sigmas[i + 1]
            d_i = (x - denoise_fn(x, _full(s_i))) / s_i
            x_next = x + (s_next - s_i) * d_i
            if i < steps - 1:
                d_next = (x_next - denoise_fn(x_next, _full(s_next))) / s_next
                x_next = x + (s_next - s_i) * 0.5 * (d_i + d_next)
            x = x_next.masked_fill(~valid, 0)
        return self.decode(x).masked_fill(~valid, 0)
