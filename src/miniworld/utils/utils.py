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


def get_inverse_sqrt_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = int(1e3),
    decay_ref_steps: int = int(7e4),
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a LambdaLR with linear warmup then inverse-sqrt decay (EDM2 Eq. 67).

    1) linearly warms up 0 -> 1 over the first ``warmup_steps``;
    2) stays at 1 until ``decay_ref_steps`` (= t_ref), then decays as
       ``1 / sqrt(step / decay_ref_steps)``.

    This is the schedule EDM2 pairs with forced weight normalization: once
    ``||w||`` is pinned the effective LR no longer self-decays via weight growth,
    so the decay must be applied explicitly. ``decay_ref_steps`` (t_ref) is the
    main knob to retune.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(warmup_steps)
        return 1.0 / float(np.sqrt(max(step / decay_ref_steps, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch Tensor to a numpy array."""
    return tensor.detach().cpu().numpy()
