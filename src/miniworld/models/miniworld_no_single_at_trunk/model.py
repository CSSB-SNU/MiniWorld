"""MiniWorld variant: the trunk does NOT carry a token_single track at all.

Differences vs ``miniworld_edm``:
  * Trunk: no ``token_single`` initialization, no ``add_single_recycle``, and
    the Pairformer runs pair-only (``use_single=False``) so it never produces a
    ``token_single_trunk``. The trunk returns only ``token_single_input`` (the
    raw per-token input embedding) and ``token_pair``.
  * Diffusion: ``DiffusionConditioning`` builds the token-single conditioning
    from ``token_single_input`` ALONE (no trunk single in the concat), and the
    atom-attention encoder is seeded with that conditioning instead of the
    (now non-existent) trunk single.

Motivation: the trunk single track was unsupervised in the distogram-only
pretraining and diverges during EDM fine-tuning (low-rank collapse → magnitude
blow-up → loss explosion). See docs/edm_token_single_rank_collapse.md. This
version removes that pathway entirely.

The training Client (param-policy, loss_fn, training loop) is identical to
``miniworld_edm`` and is reused verbatim in this package's ``client.py``.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logging

import numpy as np
import torch
from pydantic import BaseModel
from team_gm.modules import (
    DiffusionTransformer,
    MSAModule,
    Pairformer,
    SWAAtomTransformer,
)
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
    convert_linears_to_mp,
)
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.modules.diffusion_module import (
    DiffusionConditioning,
    DiffusionModule,
)
from team_gm.modules.layers import fourier_embedding
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


class DiffusionConditioningNoSingle(DiffusionConditioning):
    """DiffusionConditioning that ignores the trunk single track.

    The token-single conditioning is derived from ``token_single_input`` only,
    so ``linear_token_single`` is resized to consume ``d_single_token_input``
    (instead of ``d_single_token_input + d_single``).
    """

    def __init__(
        self,
        shared_config: SharedConfig,
        dit_cond_config: DiffusionConditioning.Config,
    ) -> None:
        super().__init__(shared_config=shared_config, dit_cond_config=dit_cond_config)
        # Consume ONLY token_single_input (no trunk single concatenation).
        self.linear_token_single = nn.Sequential(
            LayerNorm(shared_config.d_single_token_input),
            Linear(
                shared_config.d_single_token_input,
                shared_config.d_single,
                bias=False,
            ),
        )

    def forward(  # type: ignore[override]
        self,
        scheme: SchemeFeatures,
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_input"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single"],
        Float[torch.Tensor, "B L_token L_token d_pair"],
        Float[torch.Tensor, "B L_token d_single"],
    ]:
        """Forward pass; token single conditioning uses token_single_input only."""
        rel_emb = self.relative_position_embedder(
            asym_id=scheme.token_asym_id,
            token_residue_idx=scheme.token_residue_idx,
            token_idx=scheme.token_idx,
            entity_id=scheme.token_entity_id,
            sym_id=scheme.token_sym_id,
        )
        token_pair = torch.cat([token_pair_trunk, rel_emb], dim=-1)
        token_pair = self.linear_token_pair(token_pair)
        for transition in self.pair_transitions:
            token_pair = token_pair + transition(token_pair)

        token_single_input_proj = self.linear_token_single(token_single_input)
        time_embedding = fourier_embedding(t_emb)
        time_embedding = time_embedding.squeeze(-2)
        token_single = token_single_input_proj + self.add_time_embedding(time_embedding)

        for transition in self.single_transitions:
            token_single = token_single + transition(token_single)

        token_single = self.final_layernorm_token_single(token_single)
        return token_single, token_pair, token_single_input_proj


class DiffusionModuleNoSingle(DiffusionModule):
    """DiffusionModule wired for a trunk with no single track.

    Swaps in :class:`DiffusionConditioningNoSingle` and seeds the atom-attention
    encoder with the conditioning output (``token_single_cond``) rather than the
    trunk single.
    """

    def __init__(
        self,
        shared_config: SharedConfig,
        atom_dit_config: DiffusionTransformer.Config,
        token_dit_config: DiffusionTransformer.Config,
        dit_cond_config: DiffusionConditioning.Config,
        swa_atom_config: SWAAtomTransformer.Config | None = None,
    ) -> None:
        super().__init__(
            shared_config,
            atom_dit_config,
            token_dit_config,
            dit_cond_config,
            swa_atom_config=swa_atom_config,
        )
        self.diffusion_conditioning = DiffusionConditioningNoSingle(
            shared_config=shared_config,
            dit_cond_config=dit_cond_config,
        )

    def forward(  # type: ignore[override]
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module (no trunk single)."""
        (
            token_single_cond,
            token_pair_cond,
            enc_token_single,
        ) = self.diffusion_conditioning(
            scheme,
            t_emb,
            token_single_input,
            token_pair_trunk,
        )
        # The atom encoder expects a BATCH-level (B, L, d_single) per-token
        # single. Reuse the pre-time token-single projection from conditioning
        # so MPLinear forced normalization is applied once per graph.
        token_single_rep, atom_single_rep, atom_single_cond, carry = self._encode_atoms(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            enc_token_single,
            token_pair_cond,
        )
        token_single_rep = token_single_rep + self.add_single_token_cond(
            token_single_cond,
        )
        token_single_rep = self.diffusion_transformer(
            token_single_rep,
            token_single_cond,
            token_pair_cond,
            mask=structure.token_mask.unsqueeze(0).expand(
                token_single_rep.shape[0],
                -1,
                -1,
            ),
        )
        token_single_rep = self.ln_token_single_rep(token_single_rep)
        return self._decode_atoms(
            scheme,
            structure,
            token_single_rep,
            atom_single_rep,
            atom_single_cond,
            carry,
        )


class Model(nn.Module):
    """Structure AF3-like model with NO token_single track in the trunk."""

    class TrunkConfig(BaseModel):
        """Configuration for trunk modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        template_embedder: TemplatePairformer.Config
        n_recycle_max: int = 4
        use_template: bool = True
        use_contact: bool = True

    class DiffusionConfig(BaseModel):
        """Configuration for diffusion module."""

        atom_dit: DiffusionTransformer.Config
        token_dit: DiffusionTransformer.Config
        dit_cond: DiffusionConditioning.Config
        # ESMFold2/RoPE-SWA atom attention: when set, the atom encoder/decoder
        # use a sliding-window 3D-RoPE transformer with NO atom-pair tensor
        # (atom_dit is then unused for the atom path). Leave None for the
        # default AF3 atom attention. Under mp_full this path is converted too;
        # only final_denoising stays a raw zero-init Linear.
        swa_atom: SWAAtomTransformer.Config | None = None
        # EDM2 full magnitude preservation: MP every linear in the diffusion
        # module EXCEPT the final denoising (R3 output) projection. Propagates
        # mp_full to the atom/token DiTs and recursively swaps the remaining
        # diffusion linears (conditioning, projections) to MPLinear. Default off.
        mp_full: bool = False
        # Run the diffusion module forward under bf16 autocast (params stay
        # fp32 -> fp32 master weights / optimizer state for stable training).
        # The output is cast back to fp32 (EuclideanDiffuser.cal_loss requires
        # float32). Trunk is unaffected (it is manually bf16 and frozen).
        autocast_bf16: bool = False

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

        # feature initialization. produce_single_init=False: this variant never
        # uses token_single_init (no trunk single track), so we avoid creating
        # the unused ``to_token_init`` param (which would otherwise be a
        # trainable parameter that receives no gradient and trips DDP).
        self.input_feature_embedder = InputFeatureEmbedder(
            config.shared,
            config.input_feat_embbeder,
            produce_single_init=False,
        )

        # Recycle layer (pair only — no single recycle in this variant).
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

        # Diffusion module (no-single variant). Params stay fp32; bf16 is opt-in
        # via autocast at forward time (config.diffusion.autocast_bf16).
        # Full-MP: propagate mp_full into every diffusion transformer path so
        # blocks use MP linears + mp_sum + MP-SwiGLU (+ rotation if enabled).
        use_swa_atom = config.diffusion.swa_atom is not None
        if config.diffusion.mp_full:
            config.diffusion.token_dit.mp_full = True
            if use_swa_atom:
                config.diffusion.swa_atom.mp_full = True
                config.diffusion.swa_atom.magnitude_preserving = True
                config.diffusion.swa_atom.use_rotation = (
                    config.diffusion.token_dit.use_rotation
                    or config.diffusion.atom_dit.use_rotation
                )
                config.diffusion.swa_atom.mp_residual = (
                    config.diffusion.token_dit.mp_residual
                    or config.diffusion.atom_dit.mp_residual
                )
                config.diffusion.swa_atom.residual_t = config.diffusion.token_dit.residual_t
            else:
                config.diffusion.atom_dit.mp_full = True
        self.diffusion_module = DiffusionModuleNoSingle(
            config.shared,
            config.diffusion.atom_dit,
            config.diffusion.token_dit,
            config.diffusion.dit_cond,
            swa_atom_config=config.diffusion.swa_atom,
        ).to(torch.float32)
        # Full-MP: swap the remaining (non-DiT-block) diffusion linears
        # (conditioning, atom enc/dec projections) to MPLinear too, EXCLUDING the
        # final denoising R3 projection (its scale is handled by c_out/sigma_data).
        # Already-MP block linears are skipped; Transition modules are forced to
        # the PyTorch path inside the helper.
        if config.diffusion.mp_full:
            mp_exclude = ("final_denoising",)
            n_mp = convert_linears_to_mp(
                self.diffusion_module, exclude_substrings=mp_exclude
            )
            logging.getLogger(__name__).info(
                "mp_full: converted %d additional diffusion linears to MPLinear", n_mp
            )
        self.autocast_bf16 = config.diffusion.autocast_bf16

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
        """Forward pass of the condition modules with recycling (pair only)."""
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

        # input feature embedding (token_single_init is intentionally unused).
        (
            token_single_input,
            _token_single_init,
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

                # Pair-only Pairformer (use_single=False); single arg is None.
                token_pair, _ = self.pairformer_blocks.forward(
                    token_pair,
                    None,
                    token_mask,
                )
        # reduce token_pair information to distogram
        distogram_logit = self.distogram_head(token_pair)

        return (
            token_single_input,
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
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module."""
        if self.autocast_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.diffusion_module(
                    reference,
                    scheme,
                    structure,
                    x_t,
                    x_mask,
                    t_emb,
                    token_single_input,
                    token_pair_trunk,
                )
            # EuclideanDiffuser.cal_loss requires float32 x_update.
            return out.float()
        # fp32 diffusion forward: the frozen trunk emits bf16 conditioning, so
        # cast it (and x_t/t_emb) to fp32 here. The attention core still runs in
        # bf16 inside the Triton kernel (see AugmentedAttentionPairBias).
        return self.diffusion_module(
            reference,
            scheme,
            structure,
            x_t.float(),
            x_mask,
            t_emb.float(),
            token_single_input.float(),
            token_pair_trunk.float(),
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
        """Forward pass of the AF3Like model (no trunk single)."""
        (
            token_single_input,
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

        (
            token_single_input,
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
        atom_mask = self.condition["structure"].atom_mask  # (B=1, L_atom)
        x_mask = atom_mask.unsqueeze(0).expand(n_str, -1, -1)  # (A, B=1, L_atom)
        x_update = self.model.diffusion_forward(
            reference=self.condition["reference"],
            scheme=self.condition["scheme"],
            structure=self.condition["structure"],
            x_t=x_t.unsqueeze(1),  # (N_str, L, 3) -> (A=N_str, B=1, L, 3)
            x_mask=x_mask,
            t_emb=t_emb[None, None, None, None],  # broadcasts over A,B
            token_single_input=self.condition["token_single_input"],
            token_pair_trunk=self.condition["token_pair_trunk"],
        )
        return x_update.squeeze(1)  # (A=N_str, B=1, L, 3) -> (N_str, L, 3)


@dataclass
class InferenceOutput:
    """Output of the AF3Like model inference."""

    atom_pos_pred: torch.Tensor  # (B, L, 3)
    distogram_logit: torch.Tensor  # (B, L, L, n_distogram_bins)
    model_traj: np.ndarray  # (B, T, L, 3)
    inter_traj: np.ndarray  # (B, T, L, 3)
    input_traj: np.ndarray  # (B, T, L, 3)
