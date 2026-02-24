import random
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer, MSAModule, Pairformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from team_gm.utils.precision_manager import PrecisionConfig
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.data.features.batch_edge_backprop import (
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.modules.diffusion_module import (
    DiffusionConditioning,
    DiffusionModule,
)
from miniworld.modules.heads import DistogramHead
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.msa_util import init_msa


class AF3LikeModel(nn.Module):
    """Structure AF3-like model."""

    class TrunkConfig(BaseModel):
        """Configuration for trunk modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class DiffusionConfig(BaseModel):
        """Configuration for diffusion module."""

        atom_dit: DiffusionTransformer.Config
        token_dit: DiffusionTransformer.Config
        dit_cond: DiffusionConditioning.Config

    class Config(BaseModel):
        """Configuration for the AF3Like model."""

        shared: SharedConfig
        input_feat_embbeder: DiffusionTransformer.Config
        trunk: "AF3LikeModel.TrunkConfig"
        diffusion: "AF3LikeModel.DiffusionConfig"
        precision: PrecisionConfig

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
        self.add_single_recycle = nn.Sequential(
            LayerNorm(
                config.shared.d_single,
            ),
            Linear(
                config.shared.d_single,
                config.shared.d_single,
                init="zero",
            ),
        )

        # Trunk forward
        self.msa_module = MSAModule(config.trunk.msa_module)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)
        self.distogram_head = DistogramHead(
            config.shared.d_pair,
            config.shared.n_distogram_bins,
        )

        # Diffusion module
        self.diffusion_module = DiffusionModule(
            config.shared,
            config.diffusion.atom_dit,
            config.diffusion.token_dit,
            config.diffusion.dit_cond,
        )

    def condition_forward(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of the condition modules with recycling."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(
            msa,
            reference,
            scheme,
            sequence,
            structure,
        )
        residue_mask = structure.residue_mask

        token_pair = torch.zeros_like(token_pair_init)
        token_single = torch.zeros_like(token_single_init)
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                msa_feat = init_msa(
                    msa,
                    recycle_idx=i_cycle,
                    num_res_class=self.config.shared.num_res_class,
                )
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)

                token_pair = token_pair + self.msa_module(
                    msa_feat,
                    token_pair,
                    token_single_input,
                    residue_mask,
                )
                token_single = token_single_init + self.add_single_recycle(token_single)

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair,
                    token_single,
                    residue_mask,
                )
        # reduce token_pair information to distogram
        distogram_logit = self.distogram_head(token_pair)

        return (
            token_single_input,
            token_single,  # pyright: ignore[reportReturnType]
            token_pair,
            distogram_logit,
        )

    def diffusion_forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module."""
        return self.diffusion_module(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

    def forward(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
    ) -> tuple[
        Float[torch.Tensor, "B L_atom 3"],
        Float[torch.Tensor, "B L_token L_token"],
    ]:
        """Forward pass of the AF3Like model."""
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.condition_forward(
            msa,
            reference,
            scheme,
            sequence,
            structure,
        )
        # Diffusion forward
        atom_pos_update = self.diffusion_forward(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

        return atom_pos_update, distogram_logit


class AF3LikeModelWrapper(nn.Module):
    """Wrapper for AF3LikeModel to handle the input and output using solver."""

    def __init__(self, model: AF3LikeModel) -> None:
        super().__init__()
        self.conditioned_forwarded = False
        self.model = model

    def prepare_condition(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> None:
        """Prepare the model for conditioned forward pass."""
        if self.conditioned_forwarded:
            msg = "Conditioned forward is already done."
            raise ValueError(msg)

        # Load the batch and prepare the model for conditioned forward pass
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.model.condition_forward(
            msa,
            reference,
            scheme,
            sequence,
            structure,
        )
        self.conditioned_forwarded = True

        self.condition = {
            "reference": reference,
            "scheme": scheme,
            "structure": structure,
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
            "distogram_logit": distogram_logit,
        }

    def forward(
        self,
        x_t: Float[torch.Tensor, "N_str L 3"],
        t_emb: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "N_str L 3"]:
        """Forward pass of the model wrapper."""
        if not self.conditioned_forwarded:
            msg = "Conditioned forward must be called before forward pass."
            raise ValueError(msg)

        n_str = x_t.shape[0]
        x_mask = self.condition["structure"].atom_mask.repeat(n_str, 1).unsqueeze(0)
        x_update = self.model.diffusion_forward(
            reference=self.condition["reference"],
            scheme=self.condition["scheme"],
            structure=self.condition["structure"],
            x_t=x_t.unsqueeze(0),  # (B, L, 3) -> (1, B, L, 3)
            x_mask=x_mask,
            t_emb=t_emb[None, None, None, None],  # (,) -> (1, 1, 1, 1)
            token_single_input=self.condition["token_single_input"],
            token_single_trunk=self.condition["token_single_trunk"],
            token_pair_trunk=self.condition["token_pair_trunk"],
        )
        return x_update.squeeze(0)  # (1, B, L, 3) -> (B, L, 3)


@dataclass
class AF3LikeInferenceOutput:
    """Output of the AF3Like model inference."""

    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Distogram logits
    distogram_logit: torch.Tensor  # (B, L, L, 2)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray  # (B, T, L, 3)
