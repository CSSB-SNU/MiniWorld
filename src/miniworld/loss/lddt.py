import torch
import numpy as np

from team_gm.data.features_BioMol import Batch
from team_gm.utils import metrics
from jaxtyping import Float, Bool
from collections.abc import Sequence
from BioMol.utils.hierarchy import MoleculeType, PolymerType

Array = np.ndarray | torch.Tensor


    # 
