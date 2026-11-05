"""PyTorch module blocks MiniWorld builds on (moved from ``miniworld.blocks``, removed in miniworld-engine 2.4.2).

The Pairformer is shared with FoldForge and lives in ``foldforge.blocks``.
"""

from foldforge.blocks.pairformer import Pairformer, PairformerBlock

from miniworld.blocks.diffusion_transformer import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
)
from miniworld.blocks.msa_module import (
    MSAModule,
    MSAModuleBlock,
    MSAPairformer,
    MSAPairformerBlock,
)
from miniworld.blocks.swa_atom_transformer import (
    SWAAtomBlock,
    SWAAtomTransformer,
)
from miniworld.blocks.template import (
    TemplateEmbedder,
    TemplatePairformer,
    TemplatePairformerBlock,
)

__all__ = [
    "DiffusionTransformer",
    "DiffusionTransformerBlock",
    "MSAModule",
    "MSAModuleBlock",
    "MSAPairformer",
    "MSAPairformerBlock",
    "Pairformer",
    "PairformerBlock",
    "SWAAtomBlock",
    "SWAAtomTransformer",
    "TemplateEmbedder",
    "TemplatePairformer",
    "TemplatePairformerBlock",
]
