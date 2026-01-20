import random
from contextlib import ExitStack

import torch
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer
from team_gm.modules.blocks import MSAModule, Pairformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.data.features.features_biomol import Batch
from miniworld.modules.heads import ContactMapHead, DistogramHead
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.msa_util import init_msa
from miniworld.utils.precision_manager import PrecisionConfig


class ContactMapPredictionModel(nn.Module):
    """Structure Contact Map Prediction model."""

    class ConditionConfig(BaseModel):
        """Configuration for condition modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for the AF3 model."""

        shared: SharedConfig
        trunk: "ContactMapPredictionModel.ConditionConfig"
        input_feat_embbeder: DiffusionTransformer.Config
        precision: PrecisionConfig
        use_distogram: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization
        self.input_feature_embedder = InputFeatureEmbedder(
            config.shared,
            config.input_feat_embbeder,
        )

        # Recycle layers
        self.add_pair_recycle = nn.Sequential(
            LayerNorm(
                config.shared.d_pair,
            ),
            Linear(
                config.shared.d_pair,
                config.shared.d_pair,
                init="zero",
            ),
        )
        # Trunk forward
        self.msa_module = MSAModule(config.trunk.msa_module)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)

        # ContactMap prediction
        if self.config.use_distogram:
            self.final_head = DistogramHead(config.shared.d_pair, config.shared.n_distogram_bins)
        else:
            self.final_head = ContactMapHead(config.shared.d_pair)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        """Forward pass of the condition modules with recycling."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        if batch.msa.aligned_sequences.shape[1] != self.n_recycle_max:
            msg = (
                "The number of MSA sequences should match the number of recycle steps."
            )
            raise ValueError(msg)

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(
            batch.msa,
            batch.reference,
            batch.scheme,
            batch.sequence,
            batch.structure,
        )

        token_pair = torch.zeros_like(token_pair_init)
        # backprop cheating
        token_single_input = token_single_input + 0.0 * token_single_init.sum()
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())

                msa_feat = init_msa(
                    batch.msa,
                    recycle_idx=i_cycle,
                    num_res_class=self.msa_module.num_res_class,
                )
                msa_feat = msa_feat.to(token_pair.dtype)
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)
                token_pair = token_pair + self.msa_module(
                    msa_feat,
                    token_pair,
                    token_single_input,
                    batch.structure.residue_mask,
                )

                token_pair, _ = self.pairformer_blocks.forward(
                    token_pair,
                    None,
                    batch.structure.residue_mask,
                )

        return self.final_head(token_pair)

