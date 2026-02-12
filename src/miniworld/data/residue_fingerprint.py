from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from miniworld.data.mapping import ResidueMapping


def load_residue_fingerprint_table(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load residue fingerprint table aligned to ResidueMapping indices [0..31].

    Expected shape: (ResidueMapping.MAX_INDEX + 1, d_fp) == (32, d_fp)
    """
    path = Path(path)
    if path.suffix == ".pt":
        table = torch.load(path, map_location="cpu")
        if not isinstance(table, torch.Tensor):
            table = torch.as_tensor(table)
    elif path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "table" not in arr:
                msg = f"{path} is npz but missing key 'table'"
                raise ValueError(msg)
            arr = arr["table"]
        table = torch.from_numpy(arr)
    else:
        msg = f"Unsupported fingerprint file extension: {path.suffix}"
        raise ValueError(msg)

    table = table.to(dtype=dtype)

    expected_rows = ResidueMapping.MAX_INDEX + 1  # 32
    if table.ndim != 2:
        msg = f"Fingerprint table must be 2D, got shape {tuple(table.shape)}"
        raise ValueError(msg)
    if table.shape[0] != expected_rows:
        msg = f"Fingerprint table rows must be {expected_rows}, got {table.shape[0]}"
        raise ValueError(msg)

    return table