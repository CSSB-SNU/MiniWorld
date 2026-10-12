"""Historical checkpoint adapters retained for existing ESMFold2 consumers.

Canonical AF-family blocks import miniworld_engine.modules directly. These
older import paths preserve the checkpoint layouts and delta/residual contracts
of the pre-FoldForge workflows; do not substitute them into canonical blocks.
"""

from .adaln import AdaptiveLayerNorm
from .attention_pair_bias import AttentionPairBias
from .augmented_attention import AugmentedAttentionPairBias
from .conditioned_transition import ConditionedTransition
from .embeddings import RelativePositionEmbedding, fourier_embedding
from .msa_pair_weighted_averaging import MSAPairWeightedAveraging
from .outer_product import OuterProduct, OuterProductMean
from .swa_atom_attention import SWA3DRoPEAttention, build_3d_rope
from .transition import Transition
from .triangle_updates import (
    BidirectionalTriangleMultiplication,
    TriangleAttention,
    TriangleMultiplication,
)

__all__ = [
    "AdaptiveLayerNorm",
    "AttentionPairBias",
    "AugmentedAttentionPairBias",
    "BidirectionalTriangleMultiplication",
    "ConditionedTransition",
    "MSAPairWeightedAveraging",
    "RelativePositionEmbedding",
    "OuterProduct",
    "OuterProductMean",
    "SWA3DRoPEAttention",
    "Transition",
    "TriangleAttention",
    "TriangleMultiplication",
    "build_3d_rope",
    "fourier_embedding",
]
