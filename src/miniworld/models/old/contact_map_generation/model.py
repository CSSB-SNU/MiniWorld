import random
from contextlib import ExitStack
from typing import Literal

import torch
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer, MSAModule, Pairformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from team_gm.utils.precision_manager import PrecisionConfig
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.data.features.batch_edge_backprop import Batch
from miniworld.modules.embeddings import fourier_embedding
from miniworld.modules.heads import ContactMapHead, DistogramHead
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.msa_util import init_msa


class ContactMapGenerationModel(nn.Module):
    """Structure Contact Map Prediction model."""

    class ConditionConfig(BaseModel):
        """Configuration for condition modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for the AF3 model."""

        shared: SharedConfig
        trunk: "ContactMapGenerationModel.ConditionConfig"
        input_feat_embbeder: DiffusionTransformer.Config
        precision: PrecisionConfig
        use_distogram: bool = False
        contact_num_classes: int = 2

    def __init__(self, config: Config, transition_mode: Literal["absorbing", "other"]) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max
        self.transition_mode = transition_mode
        input_num_classes = config.contact_num_classes + 1 if transition_mode == "absorbing" else config.contact_num_classes

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
        self.contact_map_embedder = nn.Linear(
            input_num_classes,
            config.shared.d_pair,
            bias=False,
        )
        self.tau_proj = nn.Sequential(
            LayerNorm(config.shared.d_time),
            Linear(
                config.shared.d_time,
                config.shared.d_pair,
                bias=False,
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

    def forward(
        self,
        batch: Batch,
        noisy_contact_map: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass conditioned on noisy contact map and time embedding."""
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
        pair_mask = (
            batch.structure.residue_mask[:, :, None]
            * batch.structure.residue_mask[
                :,
                None,
                :,
            ]
        )
        contact_pair = self.contact_map_embedder(
            noisy_contact_map.to(token_pair_init.dtype),
        )
        token_pair_init = token_pair_init + contact_pair * pair_mask.unsqueeze(-1)
        tau_embed = fourier_embedding(tau).to(token_pair_init.dtype)
        tau_embed = self.tau_proj(tau_embed)
        token_pair_init = token_pair_init + tau_embed[:, None, None, :]

        # backprop cheating
        token_single_input = token_single_input + 0.0 * token_single_init.sum()
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)

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
