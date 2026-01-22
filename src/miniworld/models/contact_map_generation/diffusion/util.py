import torch


def symmetrize_pair(x: torch.Tensor) -> torch.Tensor:
    """Symmetrize pairwise tensors with channel as last dim (…, L, L, C)."""
    if x.ndim < 4:
        return x
    swapped = x.transpose(-3, -2)
    return 0.5 * (x + swapped)


def symmetrize_labels(x: torch.Tensor) -> torch.Tensor:
    """Symmetrize integer labels for pairwise matrices."""
    if x.ndim < 3:
        return x
    if x.ndim == 3:
        return torch.triu(x) + torch.triu(x, 1).transpose(-1, -2)
    return x
