"""PyTorch module blocks for Team GM.

team-gm holds the REPRESENTATIVE AF3 blocks (Pairformer, MSA module, template,
diffusion transformer, SWA atom transformer). They compose the building-block ops
from ``miniworld_engine.modules``. Model-specific / reduced variants (e.g. the pair-
only MiniPairformer, MiniMSAModule) belong to the terminal product repos, not here.
"""

from .diffusion_transformer import DiffusionTransformer, DiffusionTransformerBlock
from .msa_module import MSAModule, MSAModuleBlock, MSAPairformer, MSAPairformerBlock
from .pairformer import Pairformer, PairformerBlock
from .swa_atom_transformer import SWAAtomBlock, SWAAtomTransformer
from .template import TemplateEmbedder, TemplatePairformer, TemplatePairformerBlock

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
