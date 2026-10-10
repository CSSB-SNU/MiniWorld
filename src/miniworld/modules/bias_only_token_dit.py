"""``BiasOnlyTokenDiT`` -- the v2.0.0 token DiT: attention logits are the pair bias alone.

The v1 token DiT (``team_gm.DiffusionTransformer`` over the engine's
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

The block is the engine's ``BiasOnlyDiTBlock`` (same parameter names, so checkpoints are
unchanged): with an engine implementation it takes the engine's fused paths where they serve
the call (B200 bf16: hand-written CUDA + cuBLAS for training and for inference) and this
module's attention + ``ConditionedTransition`` otherwise.

``hoist_pair_bias`` (default off): every block's pair bias from ONE LayerNorm and ONE GEMM over
the pair (the engine's ``pair_bias_all``; ``ln_pair`` has no offset, so the per-block LayerNorm
weight folds into ``to_bias``) instead of a pair LayerNorm + ``to_bias`` in every block. Same
parameters, the same result up to rounding; it saves the per-block pair LayerNorm, its backward
and the 24 pair-gradient accumulations. The hoist runs outside the checkpointed blocks (as in
team-gm's ``DiffusionTransformer``) and only under autograd: without it (sampling) the blocks
take the pair, and the engine's inference path makes each block's softmax once per pair for
every solver step, which a per-call hoist could only repeat.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules.blocks._engine_impl import to_engine_impl
from team_gm.modules.exceptions import ImplementationType
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


class BiasOnlyAttention(nn.Module):
    """Attention whose logits are this block's projection of the pair representation.

    ``p = softmax(to_bias(ln_pair(pair)))`` is computed once per call and applied to the
    values of every augmentation. Returns the RAW delta (no residual), like
    ``AugmentedAttentionPairBias``; the layers it keeps from that block are initialised
    identically (``to_bias`` zero, ``to_gate`` gating, ``to_out`` zero, ``to_scale`` bias -2).
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
        super().__init__()
        if d_single % n_head != 0:
            msg = f"{d_single=} must be divisible by {n_head=}"
            raise ValueError(msg)
        self.n_head = n_head
        self.d_hidden = d_single // n_head
        impl = to_engine_impl(implementation)
        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond, implementation=impl)
        # No LayerNorm offset: it would only add a per-head constant to every logit, which
        # softmax cancels (v1 dropped it for the same reason). Zero init as in v1 / AF3.
        self.ln_pair = LayerNorm(d_pair, bias=False, implementation=impl)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_value = Linear(d_single, d_single, bias=False)  # v1 default (LeCun) init
        self.to_gate = Linear(d_single, d_single, bias=False, init="gating")
        self.to_out = Linear(d_single, d_single, bias=False, init="zero")
        self.to_scale = Linear(d_cond, d_single, bias=True, init="default")
        self.to_scale.bias.data.fill_(-2.0)

    @typecheck
    def attention_pattern(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "B H L L"]:
        """``softmax_n(to_bias(ln_pair(pair))[b, m, n, h])`` with masked keys at zero.

        Softmax in fp32 (the logits are the bias unscaled, as in the engine kernel); the
        caller casts to the value dtype. The mask is over keys, so no real query row is
        ever fully masked. ``bias``: this block's hoisted bias [B, H, L, L]
        (``pair_bias_all``) in place of ``to_bias(ln_pair(pair))``.
        """
        if bias is None:
            bias = self.to_bias(self.ln_pair(pair)).permute(0, 3, 1, 2)  # [B, H, L, L]
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
        return F.softmax(bias.float(), dim=-1)

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        *,
        bias: Float[torch.Tensor, "B H L L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        p = self.attention_pattern(pair, mask, bias=bias)
        x = self.ada_ln_in(single, cond)
        a, b, length, _ = x.shape
        value = self.to_value(x).view(a, b, length, self.n_head, self.d_hidden)
        # p[b,h,m,n] value[a,b,n,h,d] -> out[a,b,m,h,d]: one GEMM per (b, h); the A
        # augmentations ride along as GEMM columns (the kernel's broadcast ``t`` axis).
        out = torch.einsum("bhmn,abnhd->abmhd", p.to(value.dtype), value)
        out = out.reshape(a, b, length, self.n_head * self.d_hidden)
        out = sigmoid_gate(self.to_gate(x), out)
        out = self.to_out(out)
        return single + sigmoid_gate(self.to_scale(cond), out)


class BiasOnlyTokenDiTBlock(BiasOnlyDiTBlock):
    """AdaLN bias-only attention + engine ``ConditionedTransition``, plain residuals.

    The engine's ``BiasOnlyDiTBlock`` -- its ``forward`` takes the fused B200 paths when they
    serve the call -- with this module's parts: ``BiasOnlyAttention`` (fp32 softmax,
    ``attention_pattern``) and the transition on the configured engine backend.
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
        impl = to_engine_impl(implementation)
        super().__init__(d_single, d_cond, d_pair, n_head, implementation=impl)
        self.attention = BiasOnlyAttention(
            d_single, d_cond, d_pair, n_head, implementation=implementation,
        )
        self.transition = ConditionedTransition(
            d_hidden=d_single, d_cond=d_cond, implementation=impl,
        )


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

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            BiasOnlyTokenDiTBlock(
                config.d_single, config.d_cond, config.d_pair, config.n_head,
                implementation=config.implementation,
            )
            for _ in range(config.n_block)
        ])

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Same call signature as ``team_gm.DiffusionTransformer`` (drop-in in ``DiffusionModule``)."""
        if mask is not None and mask.ndim == 3:
            mask = mask[0]  # augment-invariant key mask
        if self.config.hoist_pair_bias and torch.is_grad_enabled():
            # one LayerNorm + one GEMM for every block's bias, OUTSIDE the checkpointed segments (team-gm's place)
            # imported here: only an engine with the hoist (89f5ab28 and later) has it; the default path keeps working
            # with older pinned engines
            from miniworld_engine.modules.bias_only_dit import pair_bias_all
            biases = pair_bias_all(self.blocks, pair)
            block_fns = [(lambda s, block=b, bias=bb: block(s, cond, None, mask, bias=bias))
                         for b, bb in zip(self.blocks, biases, strict=True)]
        else:
            block_fns = [(lambda s, block=b: block(s, cond, pair, mask)) for b in self.blocks]
        if self.config.n_checkpoint_segments is None:
            for fn in block_fns:
                single = fn(single)
            return single
        return checkpoint_sequential(block_fns, self.config.n_checkpoint_segments,
                                     input=single, use_reentrant=False)
