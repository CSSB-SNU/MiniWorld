from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer, MSAModule, Pairformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.modules.diffusion_module import (
    DiffusionConditioning,
    DiffusionModule,
)
from miniworld.modules.heads import DistogramHead
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.msa_util import (
    apply_template_dropout,
    init_contact_feat,
    init_msa,
    init_template_feat,
    init_token_single_msa,
)
from team_gm.modules import TemplateEmbedder, TemplatePairformer

if TYPE_CHECKING:
    from jaxtyping import Bool, Float

    from miniworld.data.features import (
        MSAFeatures,
        ReferenceFeatures,
        SchemeFeatures,
        SequenceFeatures,
        StructureFeatures,
        TemplateFeatures,
    )


class Model(nn.Module):
    """Structure AF3-like model."""

    class TrunkConfig(BaseModel):
        """Configuration for trunk modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        template_embedder: TemplatePairformer.Config
        n_recycle_max: int = 4
        # Trunk pair-feature toggles. Default on (backward compatible). Set
        # both False for an MSA-only trunk (e.g. when fine-tuning on a
        # distogram trunk that was trained without template/contact).
        use_template: bool = True
        use_contact: bool = True

    class DiffusionConfig(BaseModel):
        """Configuration for diffusion module."""

        atom_dit: DiffusionTransformer.Config
        token_dit: DiffusionTransformer.Config
        dit_cond: DiffusionConditioning.Config

    class Config(BaseModel):
        """Configuration for the AF3Like model."""

        shared: SharedConfig
        input_feat_embbeder: DiffusionTransformer.Config
        trunk: Model.TrunkConfig
        diffusion: Model.DiffusionConfig

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
                dtype=torch.bfloat16,
            ),
            Linear(
                config.shared.d_pair,
                config.shared.d_pair,
                init="zero",
                dtype=torch.bfloat16,
            ),
        )
        self.add_single_recycle = nn.Sequential(
            LayerNorm(
                config.shared.d_single,
                dtype=torch.bfloat16,
            ),
            Linear(
                config.shared.d_single,
                config.shared.d_single,
                init="zero",
                dtype=torch.bfloat16,
            ),
        )

        # Trunk forward
        self.use_contact = config.trunk.use_contact
        self.use_template = config.trunk.use_template
        if self.use_contact:
            self.proj_contact = Linear(
                config.shared.d_contact,
                config.shared.d_pair,
                bias=False,
                init="zero",
            ).to(torch.bfloat16)
        self.msa_module = MSAModule(config.trunk.msa_module).to(torch.bfloat16)
        if self.use_template:
            self.temp_embedder = TemplateEmbedder(
                d_pair=config.shared.d_pair,
                d_pair_template=config.shared.d_pair_template,
                template_pairformer_config=config.trunk.template_embedder,
            ).to(torch.bfloat16)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer).to(torch.bfloat16)
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
        ).to(torch.float32)

        self.rng = np.random.default_rng()
        self._forced_n_recycle: int | None = None

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def condition_forward(
        self,
        msa: MSAFeatures,
        template: TemplateFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of the condition modules with recycling."""
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

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(
            token_single_msa,
            reference,
            scheme,
            structure,
        )
        token_mask = structure.token_mask

        token_pair = torch.zeros_like(token_pair_init).to(torch.bfloat16)
        token_single = torch.zeros_like(token_single_init).to(torch.bfloat16)
        token_pair_init_bf16 = token_pair_init.to(torch.bfloat16)
        token_single_init_bf16 = token_single_init.to(torch.bfloat16)
        token_single_input_bf16 = token_single_input.to(torch.bfloat16)
        # Trunk forward with recycling
        msa_feat, msa_mask = init_msa(
            msa,
            num_res_class=self.config.shared.num_res_class,
            dtype=torch.bfloat16,
        )
        if self.use_contact:
            contact_feat = init_contact_feat(structure, dtype=torch.bfloat16)
        if self.use_template:
            template_feat = init_template_feat(template, dtype=torch.bfloat16)
            template_feat = apply_template_dropout(
                template_feat,
                self.config.trunk.template_embedder.dropout_prob,
                dtype=torch.bfloat16,
            )
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                token_pair = token_pair_init_bf16 + self.add_pair_recycle(
                    token_pair,
                )
                if self.use_contact:
                    token_pair = token_pair + self.proj_contact(contact_feat)
                if self.use_template:
                    token_pair = token_pair + self.temp_embedder(token_pair, template_feat)

                token_pair = token_pair + self.msa_module(
                    msa_feat,
                    msa_mask,
                    token_pair,
                    token_single_input_bf16,
                    token_mask,
                )
                token_single = token_single_init_bf16 + self.add_single_recycle(
                    token_single,
                )

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair,
                    token_single,
                    token_mask,
                )
        # reduce token_pair information to distogram
        distogram_logit = self.distogram_head(token_pair)

        return (
            token_single_input,
            token_single.to(torch.float32),  # pyright: ignore[reportReturnType]
            token_pair.to(torch.float32),  # pyright: ignore[reportReturnType]
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
        template: TemplateFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
    ) -> tuple[
        Float[torch.Tensor, "B L_atom 3"],
        Float[torch.Tensor, "B L_token L_token n_distogram_bins"],
    ]:
        """Forward pass of the AF3Like model."""
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.condition_forward(
            msa,
            template,
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


class ModelWrapper(nn.Module):
    """Wrapper for Model to handle the input and output using solver."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.conditioned_forwarded = False
        self.model = model

    def prepare_condition(
        self,
        msa: MSAFeatures,
        template: TemplateFeatures,
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
            template,
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
        # The model expects (A, B, L, 3) where A is the augmentation axis
        # (= n_diffusion_samples) and B is the actual batch (always 1 here).
        # Inject B=1 between A and L_atom — putting n_str in the B slot would
        # make ``num_aug, batch_size = x_t.shape[:2]`` read it as batch size,
        # which then overflows ``token_single_cond`` (shape (B=1, L_token, d))
        # via fancy indexing.
        atom_mask = self.condition["structure"].atom_mask  # (B=1, L_atom)
        x_mask = atom_mask.unsqueeze(0).expand(n_str, -1, -1)  # (A, B=1, L_atom)
        x_update = self.model.diffusion_forward(
            reference=self.condition["reference"],
            scheme=self.condition["scheme"],
            structure=self.condition["structure"],
            x_t=x_t.unsqueeze(1),  # (N_str, L, 3) -> (A=N_str, B=1, L, 3)
            x_mask=x_mask,
            t_emb=t_emb[None, None, None, None],  # (,) -> (1, 1, 1, 1), broadcasts over A,B
            token_single_input=self.condition["token_single_input"],
            token_single_trunk=self.condition["token_single_trunk"],
            token_pair_trunk=self.condition["token_pair_trunk"],
        )
        return x_update.squeeze(1)  # (A=N_str, B=1, L, 3) -> (N_str, L, 3)


@dataclass
class InferenceOutput:
    """Output of the AF3Like model inference."""

    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Distogram logits
    distogram_logit: torch.Tensor  # (B, L, L, n_distogram_bins)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray  # (B, T, L, 3)

    # Array of R/T-corrupted model input coordinates per step (pre input-scaling).
    input_traj: np.ndarray  # (B, T, L, 3)
