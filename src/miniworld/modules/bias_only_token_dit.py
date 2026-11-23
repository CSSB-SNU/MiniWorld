"""``BiasOnlyTokenDiT`` -- the v2.0.0 token DiT: attention logits are the pair bias alone.

The v1 token DiT (``miniworld.blocks.DiffusionTransformer`` over the engine's
``AugmentedAttentionPairBias``) computes ``softmax(q k^T / sqrt(d) + bias(pair)) v`` in every
block, for every augmentation: 24 blocks x 48 augments of [L, L] logits, two QK GEMMs per
block, QK-norm on top. In the phase-2 profile that is 74-76% of the diffusion step
(runs/prof_diffusion_parts).

Here the logits come from the pair representation only, the way the trunk's triangle
attention already runs (``TriangleAttention(use_self_attention=False)`` -> engine
``bias_only_attention``)::

    p    = softmax( to_bias( ln_pair(pair) ) )          [B, H, L, L]   per BLOCK (own ln_pair/to_bias)
    v    = to_value( AdaLN(single, cond) )              [A, B, L, H, D]
    out  = p @ v  ->  sigmoid(to_gate(x)) * out  ->  to_out  ->  sigmoid(to_scale(cond)) * .

Each block keeps its own ``ln_pair``/``to_bias`` (as in v1 and AF3), so the 24 blocks can
learn 24 different patterns. Two things follow from ``p`` not depending on ``single``:

* within a block, ``p`` is broadcast over the A augmentations -- the [L, L] softmax runs once
  per block instead of once per augmentation, and the QK projections and QK-norm are gone;
* per-augment, per-step information (the noisy coordinates) reaches attention only through
  the VALUE path. The trunk was trained this way; for the token DiT it is the v2.0.0 change.

Everything else is kept from the v1 block, with identical initialisation: AdaLN on the
input, ``to_value`` (default init), the value gate (``gating``), ``to_out`` (zero),
the conditioned output scale (``to_scale`` bias -2), ``to_bias`` (zero), ``ln_pair`` without
offset, the engine ``ConditionedTransition``, and the plain ``x + f(x)`` residuals, each owned
by its module (both parts return ``x + f(x)``; the block chains them).

The block is the engine's ``TokenDiTMiniWorldBiasOnly`` (same parameter names as the engine 2.x ``BiasOnlyDiTBlock``, so checkpoints are
unchanged): with an engine backend it takes the engine's fused paths (H100 / A100 / 5090 CUDA, Triton) and the PyTorch reference
otherwise; the pair-bias logits and the softmax are fp32 by default in training and inference (``docs/bias_logits.md`` of the engine).

``hoist_pair_bias`` (default off): every block's pair bias from ONE LayerNorm and ONE GEMM over
the pair (the engine's ``pair_bias_all``; ``ln_pair`` has no offset, so the per-block LayerNorm
weight folds into ``to_bias``) instead of a pair LayerNorm + ``to_bias`` in every block. Same
parameters, the same result up to rounding; it saves the per-block pair LayerNorm, its backward
and the 24 pair-gradient accumulations. The hoist runs outside the checkpointed blocks (as in
the engine's ``DiffusionTransformer``) and only under autograd: without it (sampling) the blocks
take the pair, and the engine's inference path makes each block's softmax once per pair for
every solver step, which a per-call hoist could only repeat.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from miniworld.layers.implementation import ImplementationType, to_engine_impl
from miniworld.utils.typecheck import typecheck
from miniworld_engine.modules import BiasOnlyAttention, TokenDiTMiniWorldBiasOnly
from pydantic import BaseModel
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

__all__ = ["BiasOnlyAttention", "BiasOnlyTokenDiT", "BiasOnlyTokenDiTBlock"]


class BiasOnlyTokenDiTBlock(TokenDiTMiniWorldBiasOnly):
    """AdaLN bias-only attention + engine ``ConditionedTransition``, plain residuals: the engine's ``TokenDiTMiniWorldBiasOnly``.

    ``forward(single, cond, pair, mask, bias=None)`` takes the engine's fused paths when they serve the call (H100 / A100 / 5090 CUDA,
    Triton, PyTorch reference), with the pair-bias logits and the softmax in fp32 (the engine's default; the bf16 logits mode is an
    opt-in, ``set_pair_bias_logits("bf16")``, inference only). ``implementation`` is the MiniWorld ``ImplementationType``; the engine
    receives it as its ``backend`` string (``MINIWORLD_ENGINE`` -> ``"cuda"``). A call the backend cannot serve raises
    ``BackendUnsupported`` (engine 3.x: no silent fallback).
    """

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(
            d_single,
            d_cond,
            d_pair,
            n_head,
            backend=to_engine_impl(implementation),
        )

    def attention_pattern(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B H L L"]:
        """``softmax(to_bias(ln_pair(pair)))`` of this block's attention, masked keys at zero (the reference computation)."""
        attn = self.attention
        bias = attn.to_bias(attn.ln_pair(pair)).permute(0, 3, 1, 2)  # [B, H, L, L]
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
        return F.softmax(bias.float(), dim=-1)


class BiasOnlyTokenDiT(nn.Module):
    """Token DiT whose attention logits are, per block, the projected pair bias alone."""

    class Config(BaseModel):
        """Same surface as ``DiffusionTransformer.Config`` minus ``use_qk_norm`` (no q/k)."""

        d_single: int = 768
        d_cond: int = 384
        d_pair: int = 128
        n_head: int = 16
        n_block: int = 24
        n_checkpoint_segments: int | None = None
        implementation: ImplementationType = ImplementationType.PYTORCH
        # All blocks' pair biases from one LayerNorm + one GEMM (the engine's ``pair_bias_all``), outside the
        # checkpointed blocks, under autograd only. Same parameters, same result to rounding.
        hoist_pair_bias: bool = False
        # Under autograd only: every block's attention logits in fp32 (see ``_pair_biases_fp32``), passed to the blocks as ``bias=``.
        fp32_pair_bias: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                BiasOnlyTokenDiTBlock(
                    config.d_single,
                    config.d_cond,
                    config.d_pair,
                    config.n_head,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ]
        )

    def _pair_biases_fp32(self, pair: torch.Tensor) -> list[torch.Tensor]:
        """Every block's attention logits [B, H, L, L] in fp32: ``to_bias(LN(pair))`` with the per-block LayerNorm weight folded into ``to_bias``
        (``ln_pair`` has no offset), from ONE shared LayerNorm of the pair. Autocast is off and the projection is an fp32 matmul: under bf16-mixed
        the Linear would return bf16, and the logits (1e3..2e5 here) would be rounded to a resolution of 32..1024 before the softmax."""
        dp = pair.shape[-1]
        eps = float(self.blocks[0].attention.ln_pair.eps)
        # An fp32 matmul at the process's float32 matmul precision (the training scripts' "medium": TF32 tensor cores, like the engine's own fp32 path,
        # whose pair-bias projection is a single TF32 pass): the logit only has to be kept out of a bf16 output buffer.
        with torch.autocast("cuda", enabled=False):
            xhat = F.layer_norm(pair.float(), (dp,), eps=eps)  # [B, L, L, dp], shared by the blocks
            out = []
            for block in self.blocks:
                attn = block.attention
                w = attn.to_bias.weight.float() * attn.ln_pair.weight.float()  # [H, dp]
                out.append((xhat @ w.t()).permute(0, 3, 1, 2))  # [B, H, L, L]
        return out

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Same call signature as ``miniworld.blocks.DiffusionTransformer`` (drop-in in ``DiffusionModule``)."""
        if mask is not None and mask.ndim == 3:
            mask = mask[0]  # augment-invariant key mask
        if self.config.fp32_pair_bias and torch.is_grad_enabled():
            biases = self._pair_biases_fp32(pair)
            block_fns = [
                (lambda s, block=b, bias=bb: block(s, cond, None, mask, bias=bias))
                for b, bb in zip(self.blocks, biases, strict=True)
            ]
        elif self.config.hoist_pair_bias and torch.is_grad_enabled():
            # one LayerNorm + one GEMM for every block's bias, OUTSIDE the checkpointed segments (where ``DiffusionTransformer`` puts it)
            from miniworld_engine.modules.token_dit import pair_bias_all

            biases = pair_bias_all(self.blocks, pair)
            block_fns = [
                (lambda s, block=b, bias=bb: block(s, cond, None, mask, bias=bias))
                for b, bb in zip(self.blocks, biases, strict=True)
            ]
        else:
            block_fns = [
                (lambda s, block=b: block(s, cond, pair, mask)) for b in self.blocks
            ]
        if self.config.n_checkpoint_segments is None:
            for fn in block_fns:
                single = fn(single)
            return single
        return checkpoint_sequential(
            block_fns,
            self.config.n_checkpoint_segments,
            input=single,
            use_reentrant=False,
        )
