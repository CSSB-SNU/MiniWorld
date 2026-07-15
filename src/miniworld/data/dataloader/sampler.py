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
    """Distributed weighted sampler that draws only its per-rank portion.

    Standard :class:`torch.utils.data.DistributedSampler` samples
    ``total_size = num_samples * num_replicas`` indices globally each epoch
    and then strides per rank. For weighted sampling **with replacement** that
    global-then-stride pass is wasted work — each rank's slice is
    statistically identical to sampling ``num_samples`` items directly with a
    rank-distinct seed. On a multi-source manifest (26M+ items, 8 ranks) the
    global pass otherwise materialises a ~200M-index array per rank and
    triggers OOM inside forked DataLoader workers.

    We therefore:

    * accept an explicit ``num_samples`` (per-rank count) so callers pin the
      epoch length to what they actually consume (e.g. ``train_item //
      world_size``) instead of the full catalog size;
    * seed the generator from ``(base_seed, epoch, rank)`` so each rank draws
      an independent stream (equivalent to the global-then-stride draw under
      replacement);
    * fall back to a numpy CDF + ``searchsorted`` when the catalog exceeds
      ``torch.multinomial``'s 2^24 category cap, precomputing the CDF once so
      the per-epoch cost is ``O(N + num_samples · log N)`` rather than
      ``O(N · num_samples)``.
    """

    def __init__(
        self,
        dataset: Dataset[object],
        weights: list[float],
        *,
        num_samples: int | None = None,
        replacement: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, **kwargs)
        # ``num_samples`` is per-rank. Default preserves the legacy contract
        # (each rank iterates the whole catalog) for callers that don't set it.
        self.num_samples = num_samples if num_samples is not None else len(weights)
        self.replacement = replacement
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __iter__(self) -> Iterator[int]:
        # Distinct seed per rank: with-replacement sampling is independent
        # across ranks, so we skip the global-then-stride pass entirely.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        n_categories = int(self.weights.size(0))
        if n_categories > _TORCH_MULTINOMIAL_MAX_CATEGORIES:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item())
            rng = np.random.default_rng(seed)
            weights_np = self.weights.numpy().astype(np.float64)
            weights_np /= weights_np.sum()

            if self.replacement:
                # CDF + searchsorted: precompute once, then each draw is a
                # binary search. Cheaper than np.random.choice which rebuilds
                # cumulative weights internally per call.
                cdf = np.cumsum(weights_np)
                u = rng.uniform(size=self.num_samples)
                picks = np.searchsorted(cdf, u)
            else:
                picks = rng.choice(
                    n_categories,
                    size=self.num_samples,
                    replace=False,
                    p=weights_np,
                )
            indices = picks.astype(np.int64).tolist()
        else:
            indices = torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=self.replacement,
                generator=g,
            ).tolist()

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
