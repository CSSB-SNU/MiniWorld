"""The token / AF3 atom Diffusion Transformer: a stack of the engine's ``DiTBlock`` (``modules.dit``).

Every block is the engine block (attention, conditioned transition, and the engine's fused paths where they serve the call).
This file only stacks the blocks, keeps the MiniWorld checkpoint names (``attention_pair_bias.`` -> ``attention.``) and the
activation-checkpoint options.

Removed (no live config set them; git history, branch ``migrate/dedup``): the hoisted per-stack pair bias for this stack
(``hoist_pair_bias``, ``pair_biases=``, ``prepare_pair_biases``; it re-implemented the attention outside the engine module),
the explicit sm_90 attention core (``attention_core_dtype`` ``"bf16_sm90"`` / ``"tdt"``) and ``checkpoint_keep_attention``.
The Config keeps those fields so configs that spell them out unset keep loading; they raise when switched on.
``hoist_pair_bias`` remains the switch of ``BiasOnlyTokenDiT`` (``token_dit_kind: bias_only``), which ``DiffusionModule``
reads from this Config.
"""

from collections.abc import Mapping
from typing import TypeVar

import torch
from foldforge.layers.implementation import ImplementationType, to_engine_impl
from jaxtyping import Bool, Float
from miniworld_engine.modules import AugmentedAttentionPairBias
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.utils.typecheck import typecheck
from pydantic import BaseModel
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

T = TypeVar("T")

# The precision the attention CORE runs in (the engine's ``compute_dtype``); "model" = the tensors' own.
_CORE_DTYPES: dict[str, torch.dtype | None] = {
    "model": None,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


# The attention's prefix in checkpoints written before the block became the
# engine's DiTBlock (its old attribute name).
_LEGACY = "attention_pair_bias."


def _load_legacy_names(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    *_: object,
) -> None:
    """Read ``attention_pair_bias.`` keys into ``attention``."""
    old, new = prefix + _LEGACY, prefix + "attention."
    for k in [k for k in state_dict if k.startswith(old)]:
        state_dict[new + k.removeprefix(old)] = state_dict.pop(k)


def upgrade_legacy_keys(model: nn.Module, state_dict: Mapping[str, T]) -> dict[str, T]:
    """``state_dict`` with its diffusion blocks' ``attention_pair_bias.`` keys renamed.

    Each ``DiffusionTransformerBlock``'s ``attention_pair_bias.`` keys become
    ``attention.``. ``load_state_dict`` does this through the block's load pre-hook;
    readers of a raw state dict keyed by ``named_parameters`` (the EMA shadow,
    parameter policies) call this first. Other modules' own ``attention_pair_bias``
    (the pairformer's) keep their keys.
    """
    olds = tuple(
        f"{n}.{_LEGACY}" if n else _LEGACY
        for n, m in model.named_modules()
        if isinstance(m, DiffusionTransformerBlock)
    )
    out: dict[str, T] = {}
    for k, v in state_dict.items():
        old = next((o for o in olds if k.startswith(o)), None)
        if old is not None:
            k = old.removesuffix(_LEGACY) + "attention." + k.removeprefix(old)
        out[k] = v
    return out


class DiffusionTransformerBlock(DiTBlock):
    """A single block of the Diffusion Transformer: the engine's ``DiTBlock``.

    ``AugmentedAttentionPairBias`` -> ``ConditionedTransition``, each returning its
    residual output ``x + f(x)``. Being the engine's block, a call takes its fused
    paths where they serve it and the module path otherwise.

    The attention is ``self.attention`` (the engine's name); its keys are
    ``attention.`` in ``named_parameters`` and ``state_dict`` alike, and
    ``attention_pair_bias`` is the same module under its old name. Checkpoints that
    spell the keys ``attention_pair_bias.`` still load (a load pre-hook;
    :func:`upgrade_legacy_keys` for raw state dicts).
    """

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        *,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(
            d_single,
            d_cond,
            d_pair,
            n_head,
            use_qk_norm=use_qk_norm,
            implementation=to_engine_impl(implementation),
        )
        self.register_load_state_dict_pre_hook(_load_legacy_names)

    @property
    def attention_pair_bias(self) -> AugmentedAttentionPairBias:
        """The attention (``self.attention``) under this block's original name."""
        return self.attention

    @typecheck
    def forward(  # ty: ignore[invalid-method-override] - extends DiTBlock's call
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
        core_dtype: torch.dtype | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """``DiTBlock.forward`` (each part returns its residual output). ``core_dtype`` is the attention core's
        compute dtype (the engine's ``compute_dtype``; ``None`` = the tensors' own)."""
        return DiTBlock.forward(self, single, cond, pair, mask, compute_dtype=core_dtype)


class DiffusionTransformer(nn.Module):
    """A Diffusion Transformer module."""

    class Config(BaseModel):
        """Configuration for DiffusionTransformer module."""

        d_single: int = 768
        d_cond: int = 384
        d_pair: int = 128
        n_head: int = 16
        use_qk_norm: bool = False
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 3
        n_checkpoint_segments: int | None = None
        # Main's per-block API takes precedence; legacy segmentation remains supported.
        use_checkpoint: bool = False
        # BiasOnlyTokenDiT's switch (token_dit_kind: bias_only): every block's pair bias from one LayerNorm + one GEMM.
        # This stack no longer hoists (see the module docstring): True raises for it.
        hoist_pair_bias: bool = False
        # Precision the attention CORE runs in: "model" (the tensors' own), "bf16" or "fp32" -- the engine's compute_dtype,
        # which routes a bf16 core to its sm_90 kernels where they apply. "bf16_sm90" / "tdt" were removed.
        attention_core_dtype: str = "model"
        # Removed with "bf16_sm90": True raises.
        checkpoint_keep_attention: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.attention_core_dtype not in _CORE_DTYPES:
            msg = (
                f"attention_core_dtype {config.attention_core_dtype!r}: one of {sorted(_CORE_DTYPES)} "
                "('bf16_sm90' / 'tdt' were removed; 'bf16' routes the core to the engine's sm_90 kernels where they apply)"
            )
            raise ValueError(msg)
        if config.hoist_pair_bias or config.checkpoint_keep_attention:
            msg = (
                "DiffusionTransformer: hoist_pair_bias / checkpoint_keep_attention were removed for the augmented token DiT "
                "(restore them from git history if needed); hoist_pair_bias is BiasOnlyTokenDiT's option"
            )
            raise NotImplementedError(msg)
        self.config = config

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    config.d_single,
                    config.d_cond,
                    config.d_pair,
                    config.n_head,
                    use_qk_norm=config.use_qk_norm,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "A B L d_single"],
        cond: Float[torch.Tensor, "A B L d_cond"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L d_single"]:
        """Forward pass."""
        cd = _CORE_DTYPES[self.config.attention_core_dtype]
        block_fns = [
            (lambda s, block=b: block(s, cond, pair, mask, core_dtype=cd))
            for b in self.blocks
        ]

        if self.config.use_checkpoint and torch.is_grad_enabled():
            for fn in block_fns:
                single = checkpoint(fn, single, use_reentrant=False)
        elif self.config.n_checkpoint_segments is None or not torch.is_grad_enabled():
            for fn in block_fns:
                single = fn(single)
        else:
            single = checkpoint_sequential(
                block_fns,
                self.config.n_checkpoint_segments,
                input=single,
                use_reentrant=False,
            )

        return single
