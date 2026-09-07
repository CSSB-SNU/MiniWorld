from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.data import Dataset, DistributedSampler

if TYPE_CHECKING:
    from collections.abc import Iterator

SamplingMode = Literal["iid", "sweep", "fixed"]

# Odd multiplier that decorrelates the per-cycle generator seeds from each other
# and from the member-permutation seeds.
_CYCLE_SEED_STRIDE = 1000003
_MEMBER_SEED_STRIDE = 7919


class WeightedSampler(DistributedSampler):
    """Sampler that samples indices according to given weights.

    An "epoch" here is artificial: the training loop stops after
    ``items_per_epoch`` items (``TrainConfig.train_item``) even though the
    dataset is far larger, so whatever the sampler puts at the *front* of its
    order is all that gets trained on. The mode decides how that front is
    picked.

    ``iid``
        One weighted permutation of the whole dataset per epoch, reseeded from
        ``seed + epoch``. Each epoch is an independent weighted draw, so
        consecutive epochs overlap and coverage is left to chance -- a row can
        repeat across epochs (or twice within one) while other rows are never
        drawn at all. This is the historical behaviour and remains the default.

    ``sweep``
        Round-robin over groups, without replacement across epochs. Items are
        grouped by ``groups`` (one group per cluster pair / ``edge_id``). A
        *cycle* hands out ``quota`` items per group, and consecutive epochs
        consume consecutive windows of that cycle, so nothing repeats until the
        cycle is exhausted. Within a group the rows are walked in a fixed
        random order, advancing one step per cycle, so a row cannot come back
        before every other row of its group has been used. Groups whose weight
        is zero are dropped from the pool rather than parked at the tail of the
        order.

    ``fixed``
        Cycle 0's first window, every epoch. Intended for validation, where a
        subset that changes each time makes the metric incomparable between
        epochs.

    ``sweep`` and ``fixed`` are pure functions of ``(seed, epoch)``: there is no
    progress counter to checkpoint, and every rank derives the same order, so a
    resumed run continues exactly where it stopped from the epoch number alone.
    """

    def __init__(
        self,
        dataset: Dataset[object],
        weights: list[float],
        *,
        mode: SamplingMode = "iid",
        items_per_epoch: int | None = None,
        groups: list[list[int]] | None = None,
        quota_baseline: int = 1,
        quota_ceiling: int | None = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.num_samples = len(weights)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.mode: SamplingMode = mode
        self._cycle_cache: dict[int, list[int]] = {}

        if mode == "iid":
            self.items_per_epoch = self.total_size
            return

        if items_per_epoch is None or items_per_epoch <= 0:
            msg = f"mode={mode!r} needs a positive items_per_epoch"
            raise ValueError(msg)
        self.items_per_epoch = int(items_per_epoch)
        if quota_baseline < 1:
            msg = f"quota_baseline must be >= 1, got {quota_baseline}"
            raise ValueError(msg)
        if quota_ceiling is not None and quota_ceiling < quota_baseline:
            msg = (
                f"quota_ceiling ({quota_ceiling}) must be >= quota_baseline "
                f"({quota_baseline})"
            )
            raise ValueError(msg)

        # One group per index when no grouping is given, which degenerates to a
        # plain weighted sweep over items.
        if groups is None:
            groups = [[i] for i in range(len(weights))]
        if sum(len(g) for g in groups) != len(weights):
            msg = (
                "groups must partition the weights: "
                f"{sum(len(g) for g in groups)} grouped indices for "
                f"{len(weights)} weights"
            )
            raise ValueError(msg)

        self._build_pool(groups, quota_baseline, quota_ceiling)

    def _build_pool(
        self,
        groups: list[list[int]],
        quota_baseline: int,
        quota_ceiling: int | None,
    ) -> None:
        """Precompute the per-group member order, quotas and item slots."""
        # A group's mass is what its rows share, so dropping zero-mass groups
        # drops exactly the types whose sampler weight is zero.
        gen = torch.Generator()
        member_rows: list[int] = []
        offsets: list[int] = []
        sizes: list[int] = []
        quotas: list[int] = []
        masses: list[float] = []

        for group_idx, members in enumerate(groups):
            mass = float(self.weights[members].sum())
            if mass <= 0:
                continue
            # Fixed random order for this group's rows: the traversal order is
            # arbitrary but never re-rolled, which is what keeps a row from
            # returning before its siblings have been served.
            gen.manual_seed(self.seed * _MEMBER_SEED_STRIDE + group_idx)
            perm = torch.randperm(len(members), generator=gen).tolist()

            offsets.append(len(member_rows))
            member_rows.extend(members[j] for j in perm)
            sizes.append(len(members))
            quota = max(quota_baseline, min(len(members), quota_ceiling or len(members)))
            quotas.append(quota)
            masses.append(mass)

        if not member_rows:
            msg = "every group has zero weight; nothing can be sampled"
            raise ValueError(msg)

        self._member_rows = torch.tensor(member_rows, dtype=torch.long)
        self._pair_offset = torch.tensor(offsets, dtype=torch.long)
        self._pair_size = torch.tensor(sizes, dtype=torch.long)
        self._pair_quota = torch.tensor(quotas, dtype=torch.long)

        # Item slots: group g owns quota_g slots in every cycle. `item_pair`
        # says which group a slot belongs to, `item_slot` which of that group's
        # quota_g picks it is.
        self._item_pair = torch.repeat_interleave(
            torch.arange(len(quotas), dtype=torch.long),
            self._pair_quota,
        )
        starts = torch.cumsum(self._pair_quota, dim=0) - self._pair_quota
        n_items = int(self._pair_quota.sum())
        self._item_slot = torch.arange(n_items, dtype=torch.long) - starts[
            self._item_pair
        ]
        self._item_weight = torch.tensor(masses, dtype=torch.float32)[self._item_pair]

    @property
    def cycle_length(self) -> int:
        """Items in one full cycle (sum of the per-group quotas)."""
        return int(self._item_pair.numel())

    @property
    def pool_size(self) -> int:
        """Number of groups with non-zero weight."""
        return int(self._pair_quota.numel())

    def _cycle_order(self, cycle: int) -> list[int]:
        """Return one full cycle of dataset indices, in consumption order."""
        cached = self._cycle_cache.get(cycle)
        if cached is not None:
            return cached

        gen = torch.Generator()
        gen.manual_seed(self.seed + _CYCLE_SEED_STRIDE * cycle)
        # Weighted permutation of the slots: every slot is used exactly once,
        # but a heavier group tends to land earlier, so a window shorter than a
        # cycle still follows the configured type mix.
        order = torch.multinomial(
            self._item_weight,
            self.cycle_length,
            replacement=False,
            generator=gen,
        )
        pair = self._item_pair[order]
        # The member cursor keeps advancing across cycles instead of resetting,
        # so a group's rows are exhausted before any of them is served twice.
        cursor = cycle * self._pair_quota[pair] + self._item_slot[order]
        rows = self._member_rows[
            self._pair_offset[pair] + cursor % self._pair_size[pair]
        ].tolist()

        # An epoch spans at most two cycles, so a two-entry cache is enough.
        if len(self._cycle_cache) >= 2:
            self._cycle_cache.pop(next(iter(self._cycle_cache)))
        self._cycle_cache[cycle] = rows
        return rows

    def _window(self) -> list[int]:
        """Return this epoch's global window of dataset indices."""
        if self.mode == "fixed":
            return self._cycle_order(0)[: self.items_per_epoch]

        cycle, start = divmod(self.epoch * self.items_per_epoch, self.cycle_length)
        window = self._cycle_order(cycle)[start : start + self.items_per_epoch]
        # Windows need not divide the cycle evenly; carry the remainder into the
        # next cycle so no slot is skipped at the boundary.
        while len(window) < self.items_per_epoch:
            cycle += 1
            window = window + self._cycle_order(cycle)[
                : self.items_per_epoch - len(window)
            ]
        return window

    def __iter__(self) -> Iterator[int]:
        if self.mode == "iid":
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            all_indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=False,
                generator=g,
            ).tolist()

            return iter(all_indices[self.rank : self.total_size : self.num_replicas])

        return iter(self._window()[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        if self.mode == "iid":
            return self.num_samples
        # `fixed` never spills into the next cycle, so its window is short when
        # the pool holds fewer items than an epoch asks for. `sweep` always
        # tops the window up from the following cycle.
        length = self.items_per_epoch
        if self.mode == "fixed":
            length = min(length, self.cycle_length)
        return len(range(self.rank, length, self.num_replicas))
