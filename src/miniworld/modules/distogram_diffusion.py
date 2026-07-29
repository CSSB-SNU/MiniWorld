"""EDM distogram diffusion where the TRUNK itself is the denoiser.

The distogram is treated as an L x L single-channel "image" whose pixel is the per-pair
distance BIN INDEX (0..D-1). Instead of a separate denoiser network, the trunk
(MiniMSAModule + MiniPairformer) IS the denoiser: the noised bin image is fed in through
a single Linear ENCODER (added to the trunk's pair input, in the slot recycling used to
re-inject the pair state), the trunk runs one pass, and a single Linear DECODER reads the
denoised bin image back out. EDM preconditioning (Karras et al. 2022) wraps it.

This module owns only the EDM machinery + the three thin linear maps:
  * ``encoder``     : noised bin image (1 ch) -> pair rep addition  (into the trunk)
  * ``noise_proj``  : Fourier noise-level embedding -> pair rep addition
  * ``decoder``     : trunk output pair rep -> denoised bin image (1 ch)
The model orchestrates: encode_input -> run trunk -> decode_output.

The bin image is symmetric (``D_ij == D_ji``) with a masked diagonal, so noise is drawn
symmetrically and the decoded output is symmetrised. ``sigma_data`` / ``bin_center`` MUST
be the measured std / mean of the (centred) bin image at the training crop AND bin count
(see ``scripts/measure_distogram_sigma_data.py``).
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.diffusion import EDMScheduler
from team_gm.modules.layers.embeddings import fourier_embedding
from team_gm.modules.primitives import Linear
from torch import nn


class DistogramDiffusionConfig(BaseModel):
    """Config for the EDM distogram-diffusion head (trunk-as-denoiser)."""

    # Bin-image encoding: x0 = (bin - bin_center) (raw centred bins). None -> (D-1)/2.
    # sigma_data (in ``scheduler``) is the std of x0 measured on training data.
    bin_center: float | None = None

    # Number of reverse (Heun) steps at inference — the compute/quality knob that
    # #recycles used to be.
    num_sampling_steps: int = 24

    # EDM noise schedule / preconditioning. sigma_data MUST be set from measurement.
    scheduler: EDMScheduler.EDMSchedulerConfig = EDMScheduler.EDMSchedulerConfig()


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
        """Per-sample EDM noise level (log-normal)."""
        return self.scheduler.sample_noise(b).to(device=device, dtype=torch.float32)

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
        b, length = x_t.shape[0], x_t.shape[1]
        c_in = self.scheduler.input_scale(sigma)[:, None, None]
        c_noise = self.scheduler.noise_condition(sigma)  # [B]
        x_scaled = (c_in * x_t)[..., None].to(dtype)  # [B, L, L, 1]
        fo = fourier_embedding(c_noise).to(dtype)  # [B, d_fourier]
        fo = fo[:, None, None, :].expand(b, length, length, self.d_fourier)
        inp = torch.cat([x_scaled, fo], dim=-1)  # [B, L, L, 1 + d_fourier]
        return self.encoder(inp)

    @typecheck
    def decode_output(
        self,
        token_pair: Float[torch.Tensor, "B L L C"],
        x_t: Float[torch.Tensor, "B L L"],
        sigma: Float[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B L L"]:
        """Trunk output -> denoised bin image, with EDM ``c_skip``/``c_out`` (fp32)."""
        f = _symmetrize(self.decoder(token_pair)[..., 0]).float()  # [B, L, L]
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
    ) -> tuple[torch.Tensor, dict]:
        """EDM weighted-MSE over valid off-diagonal pairs (mean over batch)."""
        length = x0.shape[1]
        eye = torch.eye(length, dtype=torch.bool, device=x0.device)
        mask = pair_mask & ~eye  # exclude the deterministic diagonal
        w = self.scheduler.loss_weight(sigma)  # [B]

        sq = (x0_hat - x0) ** 2 * mask
        denom = mask.sum(dim=(-2, -1)).clamp_min(1).to(sq.dtype)
        per_sample = w * sq.sum(dim=(-2, -1)) / denom
        loss = per_sample.mean()

        with torch.no_grad():
            rmse = (sq.sum() / mask.sum().clamp_min(1)).sqrt()
        return loss, {
            "diffusion_loss": loss.item(),
            "bin_rmse": rmse.item(),
            "sigma_mean": sigma.mean().item(),
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
    ) -> Int[torch.Tensor, "B L L"]:
        """Heun (EDM deterministic) reverse process -> predicted bin indices.

        ``denoise_fn`` runs one trunk pass (encode_input -> trunk -> decode_output).
        """
        steps = num_steps or self.config.num_sampling_steps
        sigmas = self.scheduler.sampling_time_steps(steps).to(device)

        def _full(s: torch.Tensor) -> torch.Tensor:
            return s.to(torch.float32).expand(b)

        x = _symmetric_noise(b, length, device=device, dtype=torch.float32) * sigmas[0]
        for i in range(steps):
            s_i, s_next = sigmas[i], sigmas[i + 1]
            d_i = (x - denoise_fn(x, _full(s_i))) / s_i
            x_next = x + (s_next - s_i) * d_i
            if s_next > 0:
                d_next = (x_next - denoise_fn(x_next, _full(s_next))) / s_next
                x_next = x + (s_next - s_i) * 0.5 * (d_i + d_next)
            x = x_next
        return self.decode(x)
