"""Test whether set_epoch on the original dataset is visible inside DataLoader workers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    """Minimal dataset that records its epoch in each __getitem__ call."""

    def __init__(self) -> None:
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"idx": idx, "epoch": self.epoch}


def test_same_object():
    """DataLoader.dataset IS the same object as the original dataset."""
    ds = SimpleDataset()
    dl = DataLoader(ds, batch_size=2)

    assert dl.dataset is ds, "DataLoader.dataset should be the same object"

    ds.set_epoch(5)
    assert dl.dataset.epoch == 5, "Changing ds.epoch should be reflected in dl.dataset"
    print("[PASS] test_same_object: dl.dataset is ds (same reference)")


def test_num_workers_0():
    """With num_workers=0, __getitem__ runs in the main process -> epoch is always current."""
    ds = SimpleDataset()
    dl = DataLoader(ds, batch_size=4, num_workers=0)

    # epoch=0
    batches_e0 = [b for b in dl]
    epochs_e0 = torch.cat([b["epoch"] for b in batches_e0]).tolist()
    print(f"  epoch=0 -> items saw epoch: {epochs_e0}")
    assert all(e == 0 for e in epochs_e0)

    # epoch=3
    ds.set_epoch(3)
    batches_e3 = [b for b in dl]
    epochs_e3 = torch.cat([b["epoch"] for b in batches_e3]).tolist()
    print(f"  epoch=3 -> items saw epoch: {epochs_e3}")
    assert all(e == 3 for e in epochs_e3)

    print("[PASS] test_num_workers_0: set_epoch correctly reflected")


def test_num_workers_gt0_spawn():
    """With num_workers>0 and spawn, workers get a pickle of the dataset when the
    iterator is created. So set_epoch BEFORE iterating should be visible."""
    ds = SimpleDataset()
    dl = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=False,
    )

    # epoch=0
    ds.set_epoch(0)
    batches_e0 = [b for b in dl]
    epochs_e0 = torch.cat([b["epoch"] for b in batches_e0]).tolist()
    print(f"  epoch=0 -> workers saw epoch: {epochs_e0}")
    assert all(e == 0 for e in epochs_e0), f"Expected all 0, got {epochs_e0}"

    # epoch=7 — set BEFORE creating iterator
    ds.set_epoch(7)
    batches_e7 = [b for b in dl]
    epochs_e7 = torch.cat([b["epoch"] for b in batches_e7]).tolist()
    print(f"  epoch=7 -> workers saw epoch: {epochs_e7}")
    assert all(e == 7 for e in epochs_e7), f"Expected all 7, got {epochs_e7}"

    print("[PASS] test_num_workers_gt0_spawn: set_epoch visible in spawn workers")


def test_num_workers_gt0_spawn_persistent():
    """With persistent_workers=True, workers are NOT recreated between epochs.
    The dataset in each worker is pickled ONCE. Subsequent set_epoch calls
    on the main-process dataset will NOT propagate."""
    ds = SimpleDataset()
    dl = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

    # epoch=0 — first iteration creates workers
    ds.set_epoch(0)
    batches_e0 = [b for b in dl]
    epochs_e0 = torch.cat([b["epoch"] for b in batches_e0]).tolist()
    print(f"  epoch=0 -> workers saw epoch: {epochs_e0}")
    assert all(e == 0 for e in epochs_e0), f"Expected all 0, got {epochs_e0}"

    # epoch=7 — workers already exist, they still hold the OLD dataset copy
    ds.set_epoch(7)
    batches_e7 = [b for b in dl]
    epochs_e7 = torch.cat([b["epoch"] for b in batches_e7]).tolist()
    print(f"  epoch=7 -> workers saw epoch: {epochs_e7}")

    if all(e == 7 for e in epochs_e7):
        print("[INFO] persistent_workers: epoch DID propagate (unexpected for spawn)")
    elif all(e == 0 for e in epochs_e7):
        print("[WARN] persistent_workers: epoch did NOT propagate! Workers still see epoch=0")
    else:
        print(f"[INFO] persistent_workers: mixed epochs: {epochs_e7}")

    # Cleanup persistent workers
    del dl


if __name__ == "__main__":
    print("=" * 60)
    print("1. Same object test")
    print("=" * 60)
    test_same_object()

    print()
    print("=" * 60)
    print("2. num_workers=0 test")
    print("=" * 60)
    test_num_workers_0()

    print()
    print("=" * 60)
    print("3. num_workers>0, spawn, persistent_workers=False")
    print("=" * 60)
    test_num_workers_gt0_spawn()

    print()
    print("=" * 60)
    print("4. num_workers>0, spawn, persistent_workers=True")
    print("=" * 60)
    test_num_workers_gt0_spawn_persistent()

    print()
    print("=" * 60)
    print("ALL TESTS DONE")
    print("=" * 60)
