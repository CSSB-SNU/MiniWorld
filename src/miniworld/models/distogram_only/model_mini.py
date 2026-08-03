from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.modules.heads import DistogramHead
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.mini_msa_module import MiniMSAModule
from miniworld.modules.mini_pairformer import MiniPairformer
from miniworld.modules.msa_util import (
    init_msa,
    init_token_single_msa,
)

if TYPE_CHECKING:
    from jaxtyping import Float

    from miniworld.data.features import (
        MSAFeatures,
        ReferenceFeatures,
        SchemeFeatures,
        SequenceFeatures,
        StructureFeatures,
    )


class MiniModel(nn.Module):
    """Minimal distogram-only trunk: MiniMSAModule + MiniPairformer.

    Same shape as ``distogram_only.Model`` (pair-only, no single track, no
    template, distogram loss through ``token_pair``), but the trunk uses the
    reduced blocks — ``MiniMSAModule`` (OuterProductMean + bidir trimul +
    transition) and ``MiniPairformer`` (bidir trimul + transition) — dropping
    triangle attention and the MSA self-update.
    """

    class TrunkConfig(BaseModel):
        """Configuration for trunk modules."""

        pairformer: MiniPairformer.Config
        msa_module: MiniMSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for the mini distogram-only model."""

        shared: SharedConfig
        input_feat_embbeder: DiffusionTransformer.Config
        trunk: MiniModel.TrunkConfig

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization (no single init projection — pair-only trunk)
        self.input_feature_embedder = InputFeatureEmbedder(
            config.shared,
            config.input_feat_embbeder,
            produce_single_init=False,
        )

        # Pair recycle layer (single recycle dropped — single track is gone)
        self.add_pair_recycle = nn.Sequential(
            LayerNorm(
                config.shared.d_pair,
                dtype=torch.bfloat16,
            ),
            Linear(
                config.shared.d_pair,
                config.shared.d_pair,
                init="zero",
                dtype=torch.bfloat16,
            ),
        )

        # Minimal trunk (bidir trimul + transition; MSA folded via OuterProductMean)
        self.msa_module = MiniMSAModule(config.trunk.msa_module).to(torch.bfloat16)
        self.pairformer_blocks = MiniPairformer(config.trunk.pairformer).to(
            torch.bfloat16
        )
        self.distogram_head = DistogramHead(
            config.shared.d_pair,
            config.shared.n_distogram_bins,
        )

        self.rng = np.random.default_rng()
        self._forced_n_recycle: int | None = None

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def forward(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token L_token n_distogram_bins"]:
        """Run the pair-only mini trunk with recycling and return distogram logits."""
        if self._forced_n_recycle is not None:
            n_recycle = self._forced_n_recycle
        elif self.training:
            n_recycle = self.rng.integers(1, self.n_recycle_max + 1)
        else:
            n_recycle = self.n_recycle_max

        token_single_msa = init_token_single_msa(
            msa,
            sequence,
            num_res_class=self.config.shared.num_res_class,
        )

        # input feature embedding — token_single_init is None (single-track off)
        (
            token_single_input,
            _,
            token_pair_init,
        ) = self.input_feature_embedder(
            token_single_msa,
            reference,
            scheme,
            structure,
        )
        token_mask = structure.token_mask

        token_pair = torch.zeros_like(token_pair_init).to(torch.bfloat16)
        token_pair_init_bf16 = token_pair_init.to(torch.bfloat16)
        token_single_input_bf16 = token_single_input.to(torch.bfloat16)
        # Trunk forward with recycling
        msa_feat, msa_mask = init_msa(
            msa,
            num_res_class=self.config.shared.num_res_class,
            dtype=torch.bfloat16,
        )
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                token_pair = token_pair_init_bf16 + self.add_pair_recycle(
                    token_pair,
                )

                token_pair = token_pair + self.msa_module(
                    msa_feat,
                    msa_mask,
                    token_pair,
                    token_single_input_bf16,
                    token_mask,
                    scheme.token_asym_id,
                )

                token_pair = self.pairformer_blocks(
                    token_pair,
                    token_mask,
                )
        return self.distogram_head(token_pair)
