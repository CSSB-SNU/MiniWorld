"""MiniMSAModule — a minimal MSA module (MiniWorld-owned).

Keeps the MSA→pair bridge (``OuterProductMean``) but reduces the pair stack to just
**bidirectional triangle multiplication + pair transition** (no triangle attention).
MSA is embedded once and folded into pair via OuterProductMean each block; the pair
representation is then refined by the minimal pair stack. The MSA self-update
(``MSAPairWeightedAveraging`` + MSA transition) mirrors the full ``MSAModuleBlock``
and is skipped on the last block, where the refreshed MSA would no longer be consumed.

This is a *model-specific* reduced block, so per the three-layer architecture
(``libs/team-gm/docs/ARCHITECTURE.md``) it lives in the terminal (MiniWorld). It
composes ops straight from ``miniworld_engine.modules``. Residual placement follows
the engine contract: ``OuterProductMean`` (cross-tensor) applies its residual via the
``residual=pair`` arg; ``MSAPairWeightedAveraging`` owns its residual + MSA dropout;
``Transition`` and ``BidirectionalTriangleMultiplication`` own their self-residuals
(trimul fuses ``x + drop_row(f(x))`` into the kernel epilogue). The block therefore
never re-adds ``pair + drop(...)`` itself.
"""

import torch
from jaxtyping import Bool, Float, Int
from torch.utils.checkpoint import checkpoint_sequential
from miniworld_engine.modules import (
    BidirectionalTriangleMultiplication,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
)
from miniworld_engine.modules.exceptions import ImplementationType as _EngineImpl
from pydantic import BaseModel, model_validator
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType, InvalidImplementationError
from team_gm.modules.primitives import Linear
from torch import nn
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


class MiniMSAModuleBlock(nn.Module):
    """MiniMSAModule block: OPM + MSA self-update + bidir trimul + pair transition.

    OuterProductMean (MSA→pair) + MSA self-update (MSAPairWeightedAveraging + MSA
    transition) + bidir trimul + pair transition. The MSA self-update mirrors the full
    ``MSAModuleBlock`` and is skipped on the last block (``last_block=True``), where the
    refreshed MSA would no longer be consumed.
    """

    def __init__(
        self,
        d_msa: int = 64,
        d_pair: int = 128,
        d_hidden_msa: int = 32,
        p_drop: float = 0.25,
        p_drop_msa: float = 0.15,
        *,
        mask_interchain: bool = False,
        last_block: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.last_block = last_block
        # The engine owns dispatch (MINIWORLD=auto; LN folded into the kernel) and the
        # residuals: OPM applies `residual=pair`, MSAPWA owns residual + MSA dropout,
        # trimul/Transition own their self-residuals.
        eng = _to_engine_impl(implementation)
        self.outer_product_mean = OuterProductMean(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hidden=d_hidden_msa,
            mask_interchain=mask_interchain,
        )
        if not last_block:
            # MSA self-update (pair-weighted averaging + MSA transition), as in
            # MSAModuleBlock. MSAPWA owns residual + drop_msa (p_drop_msa).
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa=d_msa,
                d_pair=d_pair,
                d_hidden=d_hidden_msa,
                p_drop=p_drop_msa,
            )
            self.transition_msa = Transition(
                d_msa,
                implementation=eng,
            )
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
        msa: Float[torch.Tensor, "B N L d_msa"],
        msa_mask: Bool[torch.Tensor, "B N"] | None,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N L d_msa"], Float[torch.Tensor, "B L L d_pair"]]:
        """Forward pass: OPM -> MSA self-update -> bidir trimul -> pair transition."""
        opm_mask = None
        if msa_mask is not None:
            opm_mask = msa_mask[:, :, None].expand(-1, -1, msa.shape[2])
            if mask is not None:
                opm_mask = opm_mask & mask[:, None, :]

        pair = self.outer_product_mean(msa, opm_mask, token_asym_id, residual=pair)
        if not self.last_block:
            msa = self.msa_pair_weighted_averaging(msa, pair, mask)  # owns residual + drop_msa
            msa = self.transition_msa(msa)
        pair = self.tri_multi(pair, mask)
        pair = self.transition_pair(pair)
        return msa, pair


class MiniMSAModule(nn.Module):
    """MiniMSAModule with multiple minimal blocks."""

    class Config(BaseModel):
        """Configuration for MiniMSAModule."""

        d_msa: int = 64
        d_pair: int = 128
        d_single_token_input: int = 441
        d_hidden_msa: int = 32
        p_drop: float = 0.25
        p_drop_msa: float = 0.15
        mask_interchain: bool = False
        implementation: ImplementationType = ImplementationType.PYTORCH
        num_res_class: int = 32
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
        self.num_res_class = config.num_res_class

        self.embed_msa = Linear(self.num_res_class + 2, config.d_msa, bias=False)
        self.single_to_msa = Linear(
            config.d_single_token_input,
            config.d_msa,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMSAModuleBlock(
                    d_msa=config.d_msa,
                    d_pair=config.d_pair,
                    d_hidden_msa=config.d_hidden_msa,
                    p_drop=config.p_drop,
                    p_drop_msa=config.p_drop_msa,
                    mask_interchain=config.mask_interchain,
                    last_block=(i == config.n_block - 1),
                    implementation=config.implementation,
                )
                for i in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        msa_mask: Bool[torch.Tensor, "B N"] | None,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        msa = msa.to(pair.dtype)
        msa = self.embed_msa(msa)
        if single is not None:
            msa = msa + self.single_to_msa(single).unsqueeze(1)

        if self.config.n_checkpoint_segments is None:
            for block in self.blocks:
                msa, pair = block(msa, msa_mask, pair, mask, token_asym_id)
        else:
            # Activation checkpointing over the MSA blocks: only segment-boundary
            # (msa, pair) are kept; each segment's internals are recomputed in backward.
            # msa_mask / mask / token_asym_id are recycle-constant, closed over here.
            def run_module(module):  # noqa: ANN001, ANN202
                def forward(inputs):  # noqa: ANN001, ANN202
                    m, p = inputs
                    return module(m, msa_mask, p, mask, token_asym_id)

                return forward

            msa, pair = checkpoint_sequential(
                [run_module(b) for b in self.blocks],
                segments=self.config.n_checkpoint_segments,
                use_reentrant=False,
                input=(msa, pair),
            )
        return pair
