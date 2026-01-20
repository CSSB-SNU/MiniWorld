import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
