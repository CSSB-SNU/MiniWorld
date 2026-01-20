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


def get_step_decay_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = int(1e3),
    decay_steps: int = int(5e4),
    decay_factor: float = 0.95,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a LambdaLR scheduler.

    1) linearly warms up from 0 → 1 over the first `warmup_steps`
    2) thereafter, multiplies the lr by `decay_factor` every `decay_steps`
    The scheduler multiplies the optimizer's base_lr by the returned factor.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # warmup: 0 -> 1
            return step / float(warmup_steps)
        # step decay: factor ** floor((step - warmup_steps) / decay_steps)
        num_decays = (step - warmup_steps) // decay_steps
        return decay_factor**num_decays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch Tensor to a numpy array."""
    return tensor.detach().cpu().numpy()
