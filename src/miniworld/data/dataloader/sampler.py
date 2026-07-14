from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, DistributedSampler

if TYPE_CHECKING:
    from collections.abc import Iterator


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
