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
from miniworld_engine.modules import BidirectionalTriangleMultiplication, Transition
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
    """A single MiniPairformer block: bidir trimul + pair transition."""

    def __init__(
        self,
        d_pair: int = 128,
        p_drop: float = 0.25,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
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

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. Each engine op owns its own residual."""
        pair = self.tri_multi(pair, mask)
        return self.transition_pair(pair)


class MiniPairformer(nn.Module):
    """A MiniPairformer stack (bidir trimul + transition per block)."""

    class Config(BaseModel):
        """Configuration for MiniPairformer."""

        d_pair: int = 128
        p_drop: float = 0.25
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 4
        n_checkpoint_segments: int | None = None

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
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if self.config.n_checkpoint_segments is None:
            for block in self.blocks:
                pair = block(pair, mask)
            return pair

        def run_module(module):  # noqa: ANN001, ANN202
            def forward(pair):  # noqa: ANN001, ANN202
                return module(pair, mask)

            return forward

        return checkpoint_sequential(
            [run_module(b) for b in self.blocks],
            segments=self.config.n_checkpoint_segments,
            use_reentrant=False,
            input=pair,
        )
