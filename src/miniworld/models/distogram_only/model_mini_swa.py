from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pydantic import BaseModel
from team_gm.modules import DiffusionTransformer, MiniMSAModule, MiniPairformer
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.configs.models import AtomSWAConfig
from miniworld.modules.heads import DistogramHead
from miniworld.modules.input_feature_embedder_esmfold2_style import (
    InputFeatureEmbedderESMFold2Style,
)
from miniworld.modules.msa_util import (
    init_msa,
    init_token_single_msa,
)
from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder

if TYPE_CHECKING:
    from jaxtyping import Float

    from miniworld.data.features import (
        MSAFeatures,
        ReferenceFeatures,
        SchemeFeatures,
        SequenceFeatures,
        StructureFeatures,
        TemplateFeatures,
    )


class TemplateEmbedderConfig(BaseModel):
    """AF3 template embedder (``AF3TemplateEmbedder``) config for the mini trunk.

    The per-template trunk is trimul-only (MiniPairformer, miniworld-kernels) — no
    triangle attention, no cuequivariance — matching the main trunk's backend.
    """

    num_channels: int = 64
    n_block: int = 2
    dropout_prob: float = 0.25
    # Output projection init: "default" (LeCun, the Protenix/OpenDDE reproduction consensus)
    # or "zero". Was MW_TEMPLATE_OUT_INIT.
    out_init: Literal["default", "zero"] = "default"
    # Per-template MiniPairformer backend. Was MW_TEMPLATE_IMPL.
    implementation: Literal["MINIWORLD_KERNELS", "PYTORCH"] = "MINIWORLD_KERNELS"


class MiniSWAModel(nn.Module):
    """Distogram-only trunk with an ESMFold2-style SWA/3D-RoPE input embedder.

    Same pair-only shape as ``distogram_only.MiniModel`` (no single track, no
    template, distogram loss through ``token_pair``), but the atom-level input
    feature embedder uses the sliding-window (or global) 3D-RoPE + QK-norm
    attention (``InputFeatureEmbedderESMFold2Style``, FlashAttention-4 on
    Hopper H100 / Blackwell B200 via ``flash_attn.cute``; falls back to flex)
    instead of the pair-bias ``DiffusionTransformer`` atom encoder. The trunk is
    the reduced ``MiniMSAModule`` (OuterProductMean + bidir trimul + transition)
    + ``MiniPairformer`` (bidir trimul + transition), which route to the cute
    (tcgen05) kernels on Hopper+.

    New-version default config (set in the model yaml): 3-block SWA atom embedder,
    4 ``MiniMSAModule`` blocks, 48 ``MiniPairformer`` blocks.
    """

    class TrunkConfig(BaseModel):
        """Configuration for trunk modules."""

        pairformer: MiniPairformer.Config
        msa_module: MiniMSAModule.Config
        n_recycle_max: int = 4
        # AF3-style template stack (adds to the pair rep before the MSA module).
        use_template: bool = False
        template_embedder: TemplateEmbedderConfig = TemplateEmbedderConfig()

    class Config(BaseModel):
        """Configuration for the SWA-embedder mini distogram-only model."""

        shared: SharedConfig
        # non-atom parts of the input feature embedder still use this config
        input_feat_embbeder: DiffusionTransformer.Config
        # ESMFold2-style atom SWA/3D-RoPE front-end (set backend="flash" for FA4,
        # a large swa_window_size for global attention)
        atom_swa: AtomSWAConfig
        trunk: MiniSWAModel.TrunkConfig

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # ESMFold2-style SWA/3D-RoPE atom input embedder (pair-only trunk -> no single init)
        self.input_feature_embedder = InputFeatureEmbedderESMFold2Style(
            config.shared,
            config.input_feat_embbeder,
            atom_swa_config=config.atom_swa,
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

        # AF3 template embedder — adds to the pair rep each recycle (uses the
        # current pair as query, so it is recomputed inside the recycle loop).
        self.use_template = config.trunk.use_template
        if self.use_template:
            te = config.trunk.template_embedder
            self.temp_embedder = AF3TemplateEmbedder(
                d_pair=config.shared.d_pair,
                num_channels=te.num_channels,
                num_res_class=config.shared.num_res_class,
                n_block=te.n_block,
                dropout_prob=te.dropout_prob,
                out_init=te.out_init,
                implementation=te.implementation,
            ).to(torch.bfloat16)

        # Minimal trunk (bidir trimul + transition; MSA folded via OuterProductMean)
        self.msa_module = MiniMSAModule(config.trunk.msa_module).to(torch.bfloat16)
        self.pairformer_blocks = MiniPairformer(config.trunk.pairformer).to(
            torch.bfloat16,
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
        template: TemplateFeatures | None = None,
    ) -> Float[torch.Tensor, "B L_token L_token n_distogram_bins"]:
        """Run the pair-only mini trunk with recycling and return distogram logits."""
        if self._forced_n_recycle is not None:
            n_recycle = self._forced_n_recycle
        elif self.training:
            n_recycle = self.rng.integers(1, self.n_recycle_max + 1)
        else:
            n_recycle = self.n_recycle_max

        (
            token_pair_init_bf16,
            token_single_input_bf16,
            msa_feat,
            msa_mask,
            token_mask,
        ) = self._embed(msa, reference, scheme, sequence, structure)

        token_pair = torch.zeros_like(token_pair_init_bf16)
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                token_pair = self._trunk_step(
                    token_pair,
                    token_pair_init_bf16,
                    token_single_input_bf16,
                    msa_feat,
                    msa_mask,
                    token_mask,
                    scheme.token_asym_id,
                    template,
                )
        return self.distogram_head(token_pair)

    def _embed(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ):
        """Input embedding done once per microbatch (before the recycle loop).

        Returns the recycle-invariant tensors the trunk step consumes, so the
        recycle loop can be split into replayable CUDA-graph steps operating on a
        static ``token_pair`` buffer.
        """
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
        msa_feat, msa_mask = init_msa(
            msa,
            num_res_class=self.config.shared.num_res_class,
            dtype=torch.bfloat16,
        )
        return (
            token_pair_init.to(torch.bfloat16),
            token_single_input.to(torch.bfloat16),
            msa_feat,
            msa_mask,
            structure.token_mask,
        )

    def _trunk_step(
        self,
        token_pair,
        token_pair_init_bf16,
        token_single_input_bf16,
        msa_feat,
        msa_mask,
        token_mask,
        token_asym_id,
        template: TemplateFeatures | None,
    ):
        """One recycle iteration of the trunk (template → MSA → Pairformer).

        Pure function of ``token_pair`` and the recycle-invariant embeddings, so a
        single capture of this step (no_grad or grad) is replayable for any recycle
        count: replay under no_grad ``n_recycle-1`` times, then once with grad.
        """
        token_pair = token_pair_init_bf16 + self.add_pair_recycle(token_pair)

        # AF3 trunk order: template → MSA → Pairformer.
        if self.use_template:
            token_pair = token_pair + self.temp_embedder(
                token_pair,
                template,
                token_asym_id,
                token_mask,
            )

        token_pair = token_pair + self.msa_module(
            msa_feat,
            msa_mask,
            token_pair,
            token_single_input_bf16,
            token_mask,
            token_asym_id,
        )

        return self.pairformer_blocks(
            token_pair,
            token_mask,
        )
