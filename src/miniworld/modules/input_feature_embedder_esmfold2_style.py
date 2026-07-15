"""ESMFold2-style input feature embedder.

This variant keeps the token-level input feature embedder contract, but replaces
atom pair-bias attention with single-only sliding-window atom attention using
3D RoPE. It is intended for the SWA/3D-RoPE MiniWorld path where the atom-level
pair tensor is not materialized.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer, SWAAtomTransformer
from team_gm.modules.layers.swa_atom_attention import build_attention_params
from team_gm.modules.primitives import Linear
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.features import (
    ReferenceFeatures,
    SchemeFeatures,
    StructureFeatures,
)
from miniworld.modules.input_embedder import InputFeatureEmbedder, init_atom_features


class ESMFold2InputAtomAttentionEncoder(nn.Module):
    """Input atom encoder without atom pair bias.

    The encoder follows the ESMFold2-style SWA path used by the diffusion atom
    encoder: atom single features condition a sliding-window transformer whose
    attention uses 3D RoPE and QK-norm instead of additive pair bias.
    """

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
        atom_swa_config: AtomSWAConfig | None = None,
    ) -> None:
        super().__init__()
        self.d_single_token = shared_config.d_single
        self.atom_swa_config = atom_swa_config or AtomSWAConfig(
            enabled=True,
            backend="sdpa",
        )
        self.to_atom_single_cond = Linear(
            6,
            shared_config.d_single_atom,
            init="default",
            bias=False,
        )
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=shared_config.d_single_atom,
                d_cond=shared_config.d_single_atom,
                n_block=diffusion_config.n_block,
                n_head=diffusion_config.n_head,
                swa_window_size=self.atom_swa_config.swa_window_size,
                expansion_ratio=self.atom_swa_config.expansion_ratio,
                n_spatial_rope_pairs_per_axis=self.atom_swa_config.n_spatial_rope_pairs_per_axis,
                n_uid_rope_pairs=self.atom_swa_config.n_uid_rope_pairs,
                spatial_rope_base_frequency=self.atom_swa_config.spatial_rope_base_frequency,
                uid_rope_base_frequency=self.atom_swa_config.uid_rope_base_frequency,
                block_style="esmfold2",
            ),
        )
        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(
                shared_config.d_single_atom,
                self.d_single_token,
                init="default",
                bias=False,
            ),
            nn.ReLU(),
        )
        self.head_dim = shared_config.d_single_atom // diffusion_config.n_head

    @typecheck
    def _scatter_atom_to_token(
        self,
        token_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "B L_atom d_single_atom"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        atom_single_rep = atom_single_rep * atom_mask[..., None]
        to_add_single_token_rep = self.atom_single_rep_to_token_single(atom_single_rep)
        token_length = int(token_idx.shape[1])
        mapping = torch.nn.functional.one_hot(
            atom_to_token_idx_map,
            num_classes=token_length,
        ).to(to_add_single_token_rep.dtype)
        token_sum = torch.einsum("bat,bad->btd", mapping, to_add_single_token_rep)
        count = torch.einsum("bat,ba->bt", mapping, atom_mask.to(mapping.dtype))
        return token_sum / count.unsqueeze(-1).clamp(min=1.0)

    @typecheck
    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        atom_single_init, _ = init_atom_features(reference)
        atom_single = self.to_atom_single_cond(atom_single_init)
        cos, sin = self.atom_transformer.build_rope(
            reference.pos, reference.space_uid,
        )
        attention_params = build_attention_params(
            cos, sin, structure.atom_mask, num_aug=1,
        )
        atom_single = self.atom_transformer(atom_single, atom_single, attention_params)
        return self._scatter_atom_to_token(
            scheme.token_idx,
            structure.atom_mask,
            scheme.atom_to_token_idx_map,
            atom_single,
        )


class InputFeatureEmbedderESMFold2Style(InputFeatureEmbedder):
    """Input feature embedder with ESMFold2-style atom SWA/3D-RoPE front-end."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
        *,
        atom_swa_config: AtomSWAConfig | None = None,
        produce_single_init: bool = True,
    ) -> None:
        super().__init__(
            shared_config,
            diffusion_config,
            produce_single_init=produce_single_init,
        )
        self.atom_attention_encoder = ESMFold2InputAtomAttentionEncoder(
            shared_config,
            diffusion_config,
            atom_swa_config,
        )
