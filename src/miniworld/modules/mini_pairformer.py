"""MiniPairformer — a minimal pair-track Pairformer (MiniWorld-owned).

Each block is just **bidirectional triangle multiplication + pair transition** (no
triangle attention, no single track). Intended as a lightweight pair-only stack.

This is a *model-specific* reduced block, so per the three-layer architecture
(``libs/team-gm/docs/ARCHITECTURE.md``) it lives in the terminal (MiniWorld), not in
team-gm. It composes ops straight from ``miniworld_engine.modules`` — the engine OWNS
the backend dispatch (picks the concrete kernel per shape+arch, LayerNorm folded into
the fused kernel) AND the residual: ``BidirectionalTriangleMultiplication`` adds its
own ``x + f(x)`` (fused into the kernel epilogue, row-dropout via ``p_drop`` applied
in-module during training) and ``Transition`` adds its own residual. So the block just
calls ``module(pair, mask)`` — no external ``pair + drop(...)`` wrapper (that would
un-fuse the residual and lose the speed win).
"""

import torch
from jaxtyping import Bool, Float
from miniworld_engine.modules import (
    AttentionPairBias,
    BidirectionalTriangleMultiplication,
    Transition,
)
from miniworld_engine.modules.exceptions import ImplementationType as _EngineImpl
from pydantic import BaseModel, model_validator
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType, InvalidImplementationError
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from typing_extensions import Self


def _to_engine_impl(impl: ImplementationType) -> _EngineImpl:
    """Map MiniWorld's public impl selector to the miniworld-engine one.

    Only PYTORCH and MINIWORLD_ENGINE are valid for the mini modules; CUEQUIVARIANCE has
    no miniworld-engine equivalent for these ops and errors (honest dispatch — no silent
    reroute).
    """
    if impl == ImplementationType.PYTORCH:
        return _EngineImpl.PYTORCH
    if impl == ImplementationType.MINIWORLD_ENGINE:
        return _EngineImpl.MINIWORLD
    raise InvalidImplementationError(impl)


class MiniPairformerBlock(nn.Module):
    """A single MiniPairformer block: bidir trimul + pair transition.

    With ``use_single=True`` it also carries a single track — AttentionPairBias
    (single self-attention biased by the pair) + single Transition, mirroring the
    full :class:`~team_gm.modules.Pairformer` block but keeping the lightweight
    pair core (bidir trimul, no triangle attention).
    """

    def __init__(
        self,
        d_pair: int = 128,
        p_drop: float = 0.25,
        *,
        d_single: int = 384,
        n_head_attention: int = 16,
        use_single: bool = False,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.use_single = use_single
        # The engine owns dispatch (MINIWORLD=auto) AND the residual: trimul folds
        # x + drop_row(f(x)) into the kernel epilogue (row-dropout via p_drop during
        # training); Transition adds its own residual. LayerNorm is folded into the
        # fused kernels.
        eng = _to_engine_impl(implementation)
        self.tri_multi = BidirectionalTriangleMultiplication(
            d_pair=d_pair,
            implementation=eng,
            p_drop=p_drop,
        )
        self.transition_pair = Transition(
            d_pair,
            implementation=eng,
        )
        if use_single:
            # AttentionPairBias owns its residual; single Transition owns its own.
            self.attention_pair_bias = AttentionPairBias(
                d_single=d_single,
                d_pair=d_pair,
                n_head=n_head_attention,
                use_qk_norm=use_qk_norm,
            )
            self.transition_single = Transition(d_single)

    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"] | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Each engine op owns its own residual.

        Returns ``pair`` when ``use_single`` is False (unchanged pair-only API),
        else ``(pair, single)``.
        """
        pair = self.tri_multi(pair, mask)
        pair = self.transition_pair(pair)
        if not self.use_single:
            return pair
        if single is None:
            msg = "single must be provided when use_single=True"
            raise ValueError(msg)
        single = self.attention_pair_bias(single, pair, mask)  # owns residual
        single = self.transition_single(single)
        return pair, single


class MiniPairformer(nn.Module):
    """A MiniPairformer stack (bidir trimul + transition per block)."""

    class Config(BaseModel):
        """Configuration for MiniPairformer."""

        d_pair: int = 128
        p_drop: float = 0.25
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 4
        n_checkpoint_segments: int | None = None
        # Optional single track (AttentionPairBias + single Transition per block).
        # When True, forward takes/returns a single of dim ``d_single`` alongside pair.
        use_single: bool = False
        d_single: int = 384
        n_head_attention: int = 16
        use_qk_norm: bool = False

        @model_validator(mode="after")
        def check_checkpoint_segments(self) -> Self:
            """Check n_checkpoint_segments is valid."""
            if self.n_checkpoint_segments is None:
                return self
            if (
                self.n_checkpoint_segments > self.n_block
                or self.n_checkpoint_segments < 1
            ):
                msg = (
                    "n_checkpoint_segments must be between 1 and n_block. "
                    f"Got n_checkpoint_segments={self.n_checkpoint_segments} "
                    f"and n_block={self.n_block}."
                )
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                MiniPairformerBlock(
                    d_pair=config.d_pair,
                    p_drop=config.p_drop,
                    d_single=config.d_single,
                    n_head_attention=config.n_head_attention,
                    use_single=config.use_single,
                    use_qk_norm=config.use_qk_norm,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"] | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Pair-only (``use_single=False``, default): takes/returns just ``pair`` — the
        original API. With ``use_single=True`` it also threads a single track and
        returns ``(pair, single)``.
        """
        use_single = self.config.use_single
        if self.config.n_checkpoint_segments is None:
            if not use_single:
                for block in self.blocks:
                    pair = block(pair, None, mask)
                return pair
            for block in self.blocks:
                pair, single = block(pair, single, mask)
            return pair, single

        if not use_single:
            def run_pair(module):  # noqa: ANN001, ANN202
                def forward(pair):  # noqa: ANN001, ANN202
                    return module(pair, None, mask)
                return forward

            return checkpoint_sequential(
                [run_pair(b) for b in self.blocks],
                segments=self.config.n_checkpoint_segments,
                use_reentrant=False,
                input=pair,
            )

        def run_both(module):  # noqa: ANN001, ANN202
            def forward(inputs):  # noqa: ANN001, ANN202
                p, s = inputs
                return module(p, s, mask)
            return forward

        pair, single = checkpoint_sequential(
            [run_both(b) for b in self.blocks],
            segments=self.config.n_checkpoint_segments,
            use_reentrant=False,
            input=(pair, single),
        )
        return pair, single
