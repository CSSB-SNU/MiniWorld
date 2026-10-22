"""ESMFold2-style input feature embedder.

This variant keeps the token-level input feature embedder contract, but replaces
atom pair-bias attention with single-only sliding-window atom attention using
3D RoPE. It is intended for the SWA/3D-RoPE MiniWorld path where the atom-level
pair tensor is not materialized.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from miniworld_engine.architecture.typing import typecheck
from miniworld_engine.modules import InputFeatureEmbedder as EngineInputFeatureEmbedder

from miniworld.configs import SharedConfig
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.features import (
    ReferenceFeatures,
    SchemeFeatures,
    StructureFeatures,
)
from miniworld.modules import DiffusionTransformer
from miniworld.modules.input_embedder import InputFeatureEmbedder


class ESMFold2InputAtomAttentionEncoder(EngineInputFeatureEmbedder):
    """Input atom encoder without atom pair bias: the engine's ``InputFeatureEmbedder`` (``miniworld_engine.modules``).

    Atom single features condition a sliding-window (or global) 3D-RoPE atom transformer -- the SWA atom DiT's fused CUDA block where
    its contract holds -- followed by an affine-free terminal RMSNorm and the mean over each token's atoms. The parameter names are
    the ones this class always had, so checkpoints load unchanged; this adapter only unpacks MiniWorld's feature containers.
    """

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
        atom_swa_config: AtomSWAConfig | None = None,
    ) -> None:
        atom_swa_config = atom_swa_config or AtomSWAConfig(enabled=True, backend="sdpa")
        super().__init__(
            shared_config.d_single_atom,
            shared_config.d_single,
            n_head=diffusion_config.n_head,
            n_block=diffusion_config.n_block,
            swa_window_size=atom_swa_config.swa_window_size,
            expansion_ratio=atom_swa_config.expansion_ratio,
            n_spatial_rope_pairs_per_axis=atom_swa_config.n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=atom_swa_config.n_uid_rope_pairs,
            spatial_rope_base_frequency=atom_swa_config.spatial_rope_base_frequency,
            uid_rope_base_frequency=atom_swa_config.uid_rope_base_frequency,
            features=atom_swa_config.input_features,
            implementation=diffusion_config.implementation,
        )
        self.d_single_token = shared_config.d_single
        self.atom_swa_config = atom_swa_config
        self.head_dim = shared_config.d_single_atom // diffusion_config.n_head

    @typecheck
    def _scatter_atom_to_token(
        self,
        token_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "B L_atom d_single_atom"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        # unbound on purpose: tests call it on a stand-in that holds only ``atom_single_rep_to_token_single``
        return EngineInputFeatureEmbedder.to_tokens(self, atom_single_rep, atom_mask, atom_to_token_idx_map, int(token_idx.shape[1]))

    @typecheck
    def forward(  # ty: ignore[invalid-method-override]  # MiniWorld's feature containers in place of the engine's tensors
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        return super().forward(
            reference.pos,
            reference.mask,
            reference.element,
            reference.charge,
            reference.space_uid,
            structure.atom_mask,
            scheme.atom_to_token_idx_map,
            int(scheme.token_idx.shape[1]),
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
