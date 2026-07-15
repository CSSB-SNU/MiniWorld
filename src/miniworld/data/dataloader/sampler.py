from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

if TYPE_CHECKING:
    from collections.abc import Iterator


# torch.multinomial rejects category counts above this — the limit shows up
# on multi-source manifests where the distillation catalog is large.
_TORCH_MULTINOMIAL_MAX_CATEGORIES = 1 << 24


class WeightedSampler(DistributedSampler):
    """Sampler that samples indices according to given weights."""

    def __init__(
        self,
        dataset: Dataset[object],
        weights: list[float],
        *,
        replacement: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.num_samples = len(weights)
        self.replacement = replacement
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        n_categories = int(self.weights.size(0))
        if n_categories > _TORCH_MULTINOMIAL_MAX_CATEGORIES:
            # torch.multinomial caps at 2^24 categories; fall back to numpy for
            # large multi-source catalogs. Seed numpy from the torch generator
            # so DDP ranks stay in sync.
            seed = int(
                torch.randint(0, 2**31 - 1, (1,), generator=g).item(),
            )
            rng = np.random.default_rng(seed)
            weights_np = self.weights.numpy().astype(np.float64)
            weights_np /= weights_np.sum()
            picks = rng.choice(
                n_categories,
                size=self.total_size,
                replace=self.replacement,
                p=weights_np,
            )
            all_indices = torch.from_numpy(picks.astype(np.int64))
        else:
            all_indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=self.replacement,
                generator=g,
            )
        perm = torch.randperm(all_indices.size(0), generator=g)
        all_indices = all_indices[perm].tolist()

        return iter(all_indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples
